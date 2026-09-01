#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import select
import tty
import termios
import shutil
import argparse
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

# ----------------------------------------------------------------------
# Color scheme handling
# ----------------------------------------------------------------------
DEFAULT_COLORS = {
    "bg": [40, 44, 52],
    "fg": [171, 178, 191],
    "red": [224, 108, 117],
    "green": [152, 195, 121],
    "yellow": [229, 192, 123],
    "blue": [97, 175, 239],
    "magenta": [198, 120, 221],
    "cyan": [86, 182, 194],
    "gutter": [75, 82, 99],
    "comment": [127, 132, 142]
}
COLORS = DEFAULT_COLORS.copy()

def load_colorscheme(name):
    global COLORS
    if not name.endswith('.json'):
        name += '.json'
    scheme_path = os.path.join("colorschemes", name)
    if not os.path.exists(scheme_path):
        print(f"Warning: colorscheme '{name}' not found. Using default.", file=sys.stderr)
        COLORS = DEFAULT_COLORS.copy()
        return COLORS
    with open(scheme_path, 'r') as f:
        colors = json.load(f)
    for key in DEFAULT_COLORS:
        if key not in colors:
            colors[key] = DEFAULT_COLORS[key]
    COLORS = colors
    return colors

# ----------------------------------------------------------------------
# Terminal utilities
# ----------------------------------------------------------------------
def set_color(fg=None, bg=None, bold=False, italic=False):
    codes = []
    if bold: codes.append("1")
    if italic: codes.append("3")
    if fg: codes.append("38;2;" + ";".join(map(str, fg)))
    if bg: codes.append("48;2;" + ";".join(map(str, bg)))
    sys.stdout.write(f"\033[{';'.join(codes)}m" if codes else "\033[0m")

def reset_color():
    sys.stdout.write("\033[0m")
    set_color(fg=COLORS["fg"], bg=COLORS["bg"])

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def move_cursor(x, y):
    sys.stdout.write(f"\033[{y};{x}H")
    sys.stdout.flush()

def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def set_raw_mode():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old

def restore_terminal_mode(old):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)

def enable_mouse():
    sys.stdout.write("\033[?1000h\033[?1002h\033[?1015h\033[?1006h")
    sys.stdout.flush()

def disable_mouse():
    sys.stdout.write("\033[?1006l\033[?1015l\033[?1002l\033[?1000l")
    sys.stdout.flush()

def read_input(timeout=0.1):
    events = []
    fd = sys.stdin.fileno()
    while True:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            break
        data = os.read(fd, 1024)
        if not data:
            break
        i = 0
        while i < len(data):
            b = data[i]
            if b == 0x1b:
                if i+1 < len(data) and data[i+1] == ord('['):
                    if i+2 < len(data) and data[i+2] == ord('<'):
                        j = i+3
                        while j < len(data) and data[j] not in (ord('M'), ord('m')):
                            j += 1
                        if j < len(data):
                            seq = data[i+3:j].decode('ascii', errors='ignore')
                            parts = seq.split(';')
                            if len(parts) == 3:
                                try:
                                    btn = int(parts[0])
                                    x = int(parts[1])
                                    y = int(parts[2])
                                    event_type = 'press' if data[j] == ord('M') else 'release'
                                    events.append(('mouse', event_type, x, y))
                                except ValueError:
                                    pass
                            i = j + 1
                            continue
                    j = i+2
                    while j < len(data) and not (0x40 <= data[j] <= 0x7e):
                        j += 1
                    if j < len(data):
                        seq = data[i:j+1].decode('ascii', errors='ignore')
                        key_map = {
                            '\033[A': 'UP',
                            '\033[B': 'DOWN',
                            '\033[C': 'RIGHT',
                            '\033[D': 'LEFT',
                            '\033[5~': 'PAGE_UP',
                            '\033[6~': 'PAGE_DOWN',
                            '\033[H': 'HOME',
                            '\033[F': 'END',
                        }
                        if seq in key_map:
                            events.append(('key', key_map[seq]))
                        i = j + 1
                        continue
                i += 1
                continue
            elif b in (0x0d, 0x0a):
                events.append(('key', 'ENTER'))
                i += 1
                continue
            elif b in (0x7f, 0x08):
                events.append(('key', 'BACKSPACE'))
                i += 1
                continue
            elif b == 0x03:
                events.append(('key', 'CTRL_C'))
                i += 1
                continue
            else:
                try:
                    events.append(('key', chr(b)))
                except:
                    pass
                i += 1
    return events

# ----------------------------------------------------------------------
# Audio file metadata extraction (filename only)
# ----------------------------------------------------------------------
def get_metadata(filepath):
    """Return (title, artist) from filename parsing."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if ' - ' in stem:
        artist, title = stem.split(' - ', 1)
        return (title.strip(), artist.strip())
    return (stem, "Unknown Artist")

# ----------------------------------------------------------------------
# Audio Player (single track)
# ----------------------------------------------------------------------
AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif', '.m4a'}

class AudioPlayer:
    def __init__(self, filepath, volume=0.7, auto_volume=False):
        self.filepath = filepath
        self.data, self.samplerate = sf.read(filepath, dtype='float32')
        if self.data.ndim > 1:
            self.data = self.data.mean(axis=1)
        self.duration = len(self.data) / self.samplerate
        self.volume = volume
        self.auto_volume = auto_volume
        # Compute normalization gain from the already loaded data
        self.normalization_gain = 1.0
        if self.auto_volume:
            peak = np.max(np.abs(self.data))
            if peak > 0:
                self.normalization_gain = min(10.0, max(1.0, 0.8 / peak))
        self.paused = False
        self.stopped = True
        self.position = 0.0
        self.stream = None
        self.fft_data = np.zeros(20)
        self.lock = threading.Lock()
        self.fft_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._fft_thread = None

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        with self.lock:
            if self.paused:
                outdata[:] = 0
                return
            start = int(self.position * self.samplerate)
            end = start + frames
            if end > len(self.data):
                outdata[:] = 0
                self.stopped = True
                self.position = self.duration
                raise sd.CallbackStop
            else:
                outdata[:] = self.data[start:end].reshape(-1, 1) * self.volume * self.normalization_gain
                self.position = end / self.samplerate

    def _fft_worker(self):
        chunk = 1024
        while not self._stop_event.is_set():
            if not self.paused and not self.stopped:
                with self.lock:
                    start = int(self.position * self.samplerate) - chunk
                    if start < 0:
                        start = 0
                    if start + chunk > len(self.data):
                        block = self.data[start:start+chunk]
                    else:
                        block = self.data[start:start+chunk]
                if len(block) == chunk:
                    window = np.hanning(chunk)
                    block = block * window
                    spectrum = np.abs(np.fft.rfft(block))
                    freqs = np.fft.rfftfreq(chunk, 1/self.samplerate)
                    mask = (freqs >= 20) & (freqs <= 20000)
                    freqs = freqs[mask]
                    spectrum = spectrum[mask]
                    if len(spectrum) > 0:
                        bands = np.logspace(np.log10(20), np.log10(20000), 21)
                        band_values = []
                        for i in range(20):
                            idx = (freqs >= bands[i]) & (freqs < bands[i+1])
                            val = np.mean(spectrum[idx]) if np.any(idx) else 0
                            band_values.append(val)
                        max_val = max(band_values) if band_values else 1
                        if max_val > 0:
                            band_values = [v / max_val for v in band_values]
                        else:
                            band_values = [0]*20
                        with self.fft_lock:
                            self.fft_data = np.array(band_values)
            time.sleep(0.05)

    def play(self):
        if self.stopped:
            self.position = 0
            self.stopped = False
            self.paused = False
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                callback=self._callback,
                blocksize=1024,
                dtype='float32'
            )
            self.stream.start()
            self._stop_event.clear()
            self._fft_thread = threading.Thread(target=self._fft_worker, daemon=True)
            self._fft_thread.start()
        elif self.paused:
            self.paused = False
            self.stream.start()

    def pause(self):
        if not self.paused and not self.stopped:
            self.paused = True
            self.stream.stop()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.stopped = True
        self.paused = False
        self._stop_event.set()
        if self._fft_thread:
            self._fft_thread.join(timeout=0.2)

    def seek(self, seconds):
        if self.stopped:
            self.play()
        with self.lock:
            new_pos = self.position + seconds
            if new_pos < 0:
                new_pos = 0
            if new_pos > self.duration:
                new_pos = self.duration
            self.position = new_pos

    def get_position(self):
        with self.lock:
            return self.position

    def set_volume(self, vol):
        self.volume = max(0.0, min(1.0, vol))

    def get_volume(self):
        return self.volume

    def get_spectrum(self):
        with self.fft_lock:
            return self.fft_data.copy()

    def cleanup(self):
        self.stop()

# ----------------------------------------------------------------------
# Playlist Manager
# ----------------------------------------------------------------------
class PlaylistManager:
    MODE_NORMAL = 'normal'
    MODE_LOOP = 'loop'
    MODE_RANDOM = 'random'

    def __init__(self, initial_files, volume=0.7, auto_volume=False,
                 crossfade_seconds=0.2):
        self.files = list(initial_files)
        self.index = 0
        self.volume = volume
        self.auto_volume = auto_volume
        self.crossfade = crossfade_seconds
        self.player = None
        self.current_file = None
        self.playlist_finished = False
        self.mode = self.MODE_NORMAL
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self._directory = None
        self._known_files = set(self.files)
        self._switching_track = False

        self._load_current()

    def _load_current(self):
        if self.player:
            self.player.cleanup()
        filepath = self.files[self.index]
        self.player = AudioPlayer(
            filepath,
            volume=self.volume,
            auto_volume=self.auto_volume
        )
        self.current_file = filepath
        self.playlist_finished = False
        self.player.play()

    def next(self):
        if self._switching_track:
            return
        if self.playlist_finished and self.mode != self.MODE_LOOP:
            return
        self._switching_track = True
        try:
            with self._lock:
                if self.mode == self.MODE_RANDOM:
                    if len(self.files) > 1:
                        new_index = self.index
                        while new_index == self.index:
                            new_index = np.random.randint(0, len(self.files))
                    else:
                        new_index = 0
                else:
                    new_index = (self.index + 1) % len(self.files)
                    if new_index == 0 and self.mode != self.MODE_LOOP and len(self.files) > 1:
                        self.playlist_finished = True
                        self.player.pause()
                        return
                self.index = new_index
            self._crossfade_to_new_track()
        finally:
            self._switching_track = False

    def previous(self):
        if self._switching_track:
            return
        self._switching_track = True
        try:
            with self._lock:
                if self.mode == self.MODE_RANDOM:
                    if len(self.files) > 1:
                        new_index = self.index
                        while new_index == self.index:
                            new_index = np.random.randint(0, len(self.files))
                    else:
                        new_index = 0
                else:
                    new_index = (self.index - 1) % len(self.files)
                self.index = new_index
            self._crossfade_to_new_track()
        finally:
            self._switching_track = False

    def _crossfade_to_new_track(self):
        if self.player:
            old_player = self.player
            if not old_player.stopped:
                # Fade out only if the old player is still playing
                fade_steps = int(self.crossfade * 100)
                for i in range(fade_steps, 0, -1):
                    old_player.set_volume(self.volume * (i / fade_steps))
                    time.sleep(0.01)
            old_player.stop()
        self._load_current()
        if self.crossfade > 0:
            fade_steps = int(self.crossfade * 100)
            for i in range(1, fade_steps + 1):
                self.player.set_volume(self.volume * (i / fade_steps))
                time.sleep(0.01)
        self.player.set_volume(self.volume)

    def check_for_track_end(self):
        if self._switching_track:
            return
        if self.player and self.player.stopped and not self.player.paused:
            if self.mode == self.MODE_LOOP or self.mode == self.MODE_RANDOM:
                self.next()
            else:
                if self.index == len(self.files) - 1:
                    self.playlist_finished = True
                else:
                    self.next()

    def cycle_mode(self):
        """Cycle through normal -> loop -> random -> normal and resume if needed."""
        if self.mode == self.MODE_NORMAL:
            self.mode = self.MODE_LOOP
        elif self.mode == self.MODE_LOOP:
            self.mode = self.MODE_RANDOM
        else:
            self.mode = self.MODE_NORMAL

        # If playback had finished and new mode allows continuous playback, restart
        if self.playlist_finished and self.mode in (self.MODE_LOOP, self.MODE_RANDOM):
            self.playlist_finished = False
            if self.player and (self.player.stopped or self.player.paused):
                self.next()
            else:
                self.player.play()

    def set_volume(self, vol):
        self.volume = vol
        if self.player:
            self.player.set_volume(vol)

    def get_current_file(self):
        return self.current_file

    def get_position(self):
        return self.player.get_position() if self.player else 0.0

    def get_duration(self):
        return self.player.duration if self.player else 0.0

    def get_spectrum(self):
        return self.player.get_spectrum() if self.player else np.zeros(20)

    def seek(self, seconds):
        if self.player:
            self.player.seek(seconds)

    def pause(self):
        if self.player:
            self.player.pause()

    def play(self):
        if self.player:
            self.player.play()

    def cleanup(self):
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        if self.player:
            self.player.cleanup()

    def start_directory_monitor(self, directory):
        self._directory = directory
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self):
        while not self._stop_monitor.is_set():
            if self._directory:
                try:
                    current_files = set()
                    for f in os.listdir(self._directory):
                        full = os.path.join(self._directory, f)
                        if os.path.isfile(full) and os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                            current_files.add(full)
                    new_files = current_files - self._known_files
                    if new_files:
                        with self._lock:
                            self.files.extend(sorted(new_files))
                            self._known_files.update(new_files)
                except Exception:
                    pass
            time.sleep(2.0)

    def get_adjacent_tracks(self):
        if not self.files:
            return (None, None)
        prev_idx = (self.index - 1) % len(self.files)
        next_idx = (self.index + 1) % len(self.files)
        return (self.files[prev_idx], self.files[next_idx])

# ----------------------------------------------------------------------
# TUI class
# ----------------------------------------------------------------------
class MusicPlayerTUI:
    def __init__(self, files_or_folder, volume=0.7, auto_volume=False):
        self.auto_volume = auto_volume
        self.running = True
        self.term_width, self.term_height = shutil.get_terminal_size()
        self.button_zones = {}
        self.progress_zone = None
        self.volume_zone = None
        self.old_settings = None

        if os.path.isdir(files_or_folder):
            self.directory = files_or_folder
            files = self._get_audio_files_in_dir(files_or_folder)
        else:
            self.directory = None
            files = [files_or_folder]

        self.playlist = PlaylistManager(
            files,
            volume=volume,
            auto_volume=auto_volume
        )
        if self.directory:
            self.playlist.start_directory_monitor(self.directory)

    def _get_audio_files_in_dir(self, directory):
        files = []
        for f in os.listdir(directory):
            full = os.path.join(directory, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                files.append(full)
        return sorted(files)

    def setup_terminal(self):
        self.old_settings = set_raw_mode()
        hide_cursor()
        enable_mouse()
        set_color(bg=COLORS["bg"])
        clear_screen()

    def restore_terminal(self):
        reset_color()
        disable_mouse()
        show_cursor()
        if self.old_settings:
            restore_terminal_mode(self.old_settings)
        clear_screen()

    def draw(self):
        self.term_width, self.term_height = shutil.get_terminal_size()
        w, h = self.term_width, self.term_height
        set_color(bg=COLORS["bg"])
        clear_screen()

        filepath = self.playlist.get_current_file()
        if filepath:
            title, artist = get_metadata(filepath)
            title = title.upper()
        else:
            title = "No track"
            artist = ""

        prev_file, next_file = self.playlist.get_adjacent_tracks()
        prev_title = os.path.splitext(os.path.basename(prev_file))[0] if prev_file else ""
        next_title = os.path.splitext(os.path.basename(next_file))[0] if next_file else ""

        max_side = max(10, min(20, (w - len(title) - 4) // 2))
        prev_trunc = self._truncate(prev_title, max_side)
        next_trunc = self._truncate(next_title, max_side)
        center_width = len(title) + 2

        move_cursor(1, 2)
        set_color(fg=COLORS["comment"], bg=COLORS["bg"])
        sys.stdout.write(prev_trunc.rjust(max_side))
        reset_color()

        move_cursor(max(1, (w - center_width)//2), 2)
        set_color(fg=COLORS["fg"], bold=True, bg=COLORS["bg"])
        sys.stdout.write(title)
        reset_color()

        move_cursor(w - max_side - 1, 2)
        set_color(fg=COLORS["comment"], bg=COLORS["bg"])
        sys.stdout.write(next_trunc.ljust(max_side))
        reset_color()

        move_cursor(max(1, (w - len(artist))//2), 3)
        set_color(fg=COLORS["comment"], italic=True, bg=COLORS["bg"])
        sys.stdout.write(artist)
        reset_color()

        buttons_y = max(11, h - 6)
        panel_top = buttons_y - 2
        available_height = panel_top - 5
        spec_height = min(10, available_height)
        spec_y = panel_top - spec_height
        spectrum = self.playlist.get_spectrum()
        heights = [int(v * spec_height) for v in spectrum]
        total_width = w - 4
        num_bars = 20
        bar_width = max(1, (total_width - (num_bars - 1)) // num_bars)
        x = 2
        for hgt in heights:
            for dy in range(spec_height - hgt, spec_height):
                move_cursor(x, spec_y + dy)
                set_color(fg=COLORS["cyan"], bg=COLORS["bg"])
                sys.stdout.write("█" * bar_width)
                reset_color()
            x += bar_width + 1

        for y in range(panel_top, h + 1):
            move_cursor(1, y)
            set_color(bg=COLORS["gutter"])
            sys.stdout.write(" " * w)
            reset_color()

        w_prev = 3; w_back = 3; w_play = 7; w_fwd = 3; w_next = 3
        gap = 2
        total = w_prev + gap + w_back + gap + w_play + gap + w_fwd + gap + w_next
        offset = max(1, (w - total)//2)
        x_prev = offset
        x_back = x_prev + w_prev + gap
        x_play = x_back + w_back + gap
        x_fwd = x_play + w_play + gap
        x_next = x_fwd + w_fwd + gap

        self._draw_icon_button(x_prev, buttons_y, "⏮", "prev")
        self._draw_icon_button(x_back, buttons_y, "↶", "back5")
        self._draw_play_pause(x_play, buttons_y-1)
        self._draw_icon_button(x_fwd, buttons_y, "↷", "forward5")
        self._draw_icon_button(x_next, buttons_y, "⏭", "next")

        mode_text = self.playlist.mode.capitalize()
        mode_btn = f" Mode: {mode_text} "
        mode_x = w - len(mode_btn) - 2
        mode_y = buttons_y + 4
        self._draw_mode_button(mode_x, mode_y, mode_btn)

        prog_y = buttons_y + 2
        pos = self.playlist.get_position()
        dur = self.playlist.get_duration()
        t_cur = self._format_time(pos)
        t_total = self._format_time(dur)
        bar_width = max(5, w - len(t_cur) - len(t_total) - 4)
        x_time_cur = 1
        x_bar = x_time_cur + len(t_cur) + 1
        x_time_total = x_bar + bar_width + 1

        move_cursor(x_time_cur, prog_y)
        set_color(fg=COLORS["comment"], bg=COLORS["gutter"])
        sys.stdout.write(t_cur)
        reset_color()

        move_cursor(x_bar, prog_y)
        set_color(fg=COLORS["fg"], bg=COLORS["gutter"])
        sys.stdout.write("─" * bar_width)
        ratio = pos / dur if dur > 0 else 0.0
        filled = int(bar_width * ratio)
        if filled > 0:
            move_cursor(x_bar, prog_y)
            set_color(fg=COLORS["blue"], bg=COLORS["gutter"], bold=True)
            sys.stdout.write("━" * filled)
        reset_color()

        move_cursor(x_time_total, prog_y)
        set_color(fg=COLORS["comment"], bg=COLORS["gutter"])
        sys.stdout.write(t_total)
        reset_color()

        self.progress_zone = (x_bar, prog_y, x_bar + bar_width - 1, prog_y)

        vol_y = prog_y + 2
        vol_label = "Volume:"
        vol_bar_x = len(vol_label) + 2
        vol_bar_width = 15
        vol_filled = int(self.playlist.volume * vol_bar_width)

        move_cursor(1, vol_y)
        set_color(fg=COLORS["comment"], bg=COLORS["gutter"])
        sys.stdout.write(vol_label)
        reset_color()

        move_cursor(vol_bar_x, vol_y)
        set_color(fg=COLORS["fg"], bg=COLORS["fg"])
        sys.stdout.write(" " * vol_bar_width)
        if vol_filled > 0:
            move_cursor(vol_bar_x, vol_y)
            set_color(fg=COLORS["red"], bg=COLORS["gutter"], bold=True)
            sys.stdout.write("█" * vol_filled)
        reset_color()

        move_cursor(vol_bar_x + vol_bar_width + 2, vol_y)
        set_color(fg=COLORS["fg"], bg=COLORS["gutter"])
        sys.stdout.write(f"{self.playlist.volume:.0%}")
        if self.auto_volume:
            sys.stdout.write(" [Auto]")
        reset_color()

        self.volume_zone = (vol_bar_x, vol_y, vol_bar_x + vol_bar_width - 1, vol_y)

        sys.stdout.flush()

    def _truncate(self, text, max_len):
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text

    def _draw_icon_button(self, x, y, symbol, button_id):
        text = f" {symbol} "
        move_cursor(x, y)
        set_color(fg=COLORS["bg"], bg=COLORS["blue"], bold=True)
        sys.stdout.write(text)
        reset_color()
        self.button_zones[button_id] = (x, y, x + len(text) - 1, y)

    def _draw_play_pause(self, x, y):
        w = 7
        move_cursor(x, y)
        set_color(fg=COLORS["blue"], bg=COLORS["gutter"], bold=True)
        sys.stdout.write("╭─────╮")
        move_cursor(x, y+1)
        is_playing = not self.playlist.player.paused and not self.playlist.player.stopped
        icon = "⏸" if is_playing else "▶"
        sys.stdout.write(f"│  {icon}  │")
        move_cursor(x, y+2)
        sys.stdout.write("╰─────╯")
        reset_color()
        self.button_zones['play_pause'] = (x, y, x + w - 1, y + 2)

    def _draw_mode_button(self, x, y, text):
        move_cursor(x, y)
        set_color(fg=COLORS["yellow"], bold=True, bg=COLORS["gutter"])
        sys.stdout.write(text)
        reset_color()
        self.button_zones['mode'] = (x, y, x + len(text) - 1, y)

    def _format_time(self, seconds):
        seconds = int(seconds)
        return f"{seconds//60:02d}:{seconds%60:02d}"

    def handle_key(self, key):
        if key == ' ':
            self.toggle_play_pause()
        elif key in ('q', 'Q', 'CTRL_C'):
            self.running = False
        elif key == 'LEFT':
            self.playlist.seek(-5)
        elif key == 'RIGHT':
            self.playlist.seek(5)
        elif key == '+':
            self.playlist.set_volume(min(1.0, self.playlist.volume + 0.05))
        elif key == '-':
            self.playlist.set_volume(max(0.0, self.playlist.volume - 0.05))
        elif key in ('m', 'M'):
            self.playlist.cycle_mode()
        elif key in ('n', 'N'):
            self.playlist.next()
        elif key in ('p', 'P'):
            self.playlist.previous()

    def handle_mouse(self, event_type, x, y):
        if event_type != 'press':
            return
        for bid, (x1, y1, x2, y2) in self.button_zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if bid == "play_pause":
                    self.toggle_play_pause()
                elif bid == "back5":
                    self.playlist.seek(-5)
                elif bid == "forward5":
                    self.playlist.seek(5)
                elif bid == "prev":
                    self.playlist.previous()
                elif bid == "next":
                    self.playlist.next()
                elif bid == "mode":
                    self.playlist.cycle_mode()
                return
        if self.progress_zone:
            x1, y1, x2, y2 = self.progress_zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                ratio = (x - x1) / (x2 - x1) if x2 > x1 else 0
                new_pos = ratio * self.playlist.get_duration()
                if self.playlist.player:
                    with self.playlist.player.lock:
                        self.playlist.player.position = new_pos
                return
        if self.volume_zone:
            x1, y1, x2, y2 = self.volume_zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                ratio = (x - x1) / (x2 - x1) if x2 > x1 else 0
                self.playlist.set_volume(ratio)
                return

    def toggle_play_pause(self):
        if self.playlist.player.paused:
            self.playlist.play()
        else:
            self.playlist.pause()

    def run(self):
        self.setup_terminal()
        try:
            while self.running:
                events = read_input(timeout=0.05)
                for ev in events:
                    if ev[0] == 'key':
                        self.handle_key(ev[1])
                    elif ev[0] == 'mouse':
                        self.handle_mouse(ev[1], ev[2], ev[3])
                self.playlist.check_for_track_end()
                self.draw()
                time.sleep(0.05)
        finally:
            self.restore_terminal()
            self.playlist.cleanup()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Terminal audio player with spectrum, playlists, and color schemes.")
    parser.add_argument("path", nargs='?', help="Audio file or directory (defaults to user's default folder)")
    parser.add_argument("-v", "--volume", type=float, default=0.7, help="Initial volume (0.0 to 1.0)")
    parser.add_argument("-a", "--auto", action="store_true", help="Enable automatic volume (AGC)")
    args = parser.parse_args()

    userdata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "userdata.json")
    userdata = {}
    if os.path.exists(userdata_path):
        with open(userdata_path, 'r') as f:
            userdata = json.load(f)

    colorscheme_name = userdata.get("colorscheme", "atom-one-dark")
    load_colorscheme(colorscheme_name)

    if args.path:
        path = args.path
    else:
        path = userdata.get("default_folder", None)
        if not path or not os.path.exists(path):
            print("Error: no valid path provided and no default_folder in userdata.json.")
            sys.exit(1)

    if not os.path.exists(path):
        print(f"Error: path '{path}' does not exist.")
        sys.exit(1)

    MusicPlayerTUI(path, volume=args.volume, auto_volume=args.auto).run()

if __name__ == "__main__":
    main()