import os
import pyaudio
import speech_recognition as sr
import tkinter as tk
from tkinter import filedialog, Listbox
from tkinter import ttk
import time
import threading
import pickle
import cv2
import mediapipe as mp
import numpy as np

from PIL import Image, ImageTk

import eyed3
import pygame
from speech_recognition.exceptions import WaitTimeoutError

from search import search_song, add_song_to_dict
from static import *
from voice import parse_voice_command

eyed3.log.setLevel("ERROR")

playlists = []

# Initialize Pygame and Tkinter
pygame.mixer.init()
root = tk.Tk()
root.title("Echo Tunes")

# Set window size and background color
root.geometry('920x600')
root.configure(bg='#7B2869')

# Create title label
title_label = tk.Label(root, text="Echo Tunes", bg='#7B2869', font=("Helvetica", 20, "bold"))
title_label.place(x=100, y=10, width=720, height=50)

# Create frames for playlist and controls
playlist_frame = tk.Frame(root, bg='#CFC7F8')
playlist_frame.place(x=300, y=185, width=590, height=225)

playlist_control_frame = tk.Frame(root, bg='#ffffff')
playlist_control_frame.place(x=300, y=145, width=590, height=40)

control_frame = tk.Frame(root, bg='#CFC7F8')
control_frame.place(x=35, y=410, width=855, height=100)

# Create a white frame for the left-side white square
white_frame = tk.Frame(root, bg='#ffffff')
white_frame.place(x=35, y=60, width=260, height=350)

# Create Treeview with four columns
columns = ('#1', '#2', '#3', '#4')
playlist = ttk.Treeview(playlist_frame, columns=columns, show='headings')
playlist.heading('#1', text='#')
playlist.heading('#2', text='Song')
playlist.heading('#3', text='Artist')
playlist.heading('#4', text='Album')
playlist.column('#1', width=40)
playlist.pack(fill=tk.BOTH, expand=True)

# Create status label for Voice Command Listening
voice_command_label = tk.Label(root, text="Voice Command: Listening...", bg='#ffffff')
voice_command_label.place(x=0, y=580, width=920, height=20)

# Create status label and volume control
status_label = tk.Label(root, text="Status: Idle", bg='#CFC7F8')
status_label.place(x=0, y=560, width=920, height=20)

# add image left white space
img_1 = Image.open('icons/5.jpg')
img_1 = img_1.resize((260, 270))
img_1 = ImageTk.PhotoImage(img_1)
app_image = tk.Label(white_frame, height=350, image=img_1, bg='#ffffff')
app_image.place(x=-2, y=-2)


# Function to update the songs list when a playlist is selected
def update_songs(evt):
    # Get the selected playlist
    selected_playlist = playlists[playlists_listbox.curselection()[0]]
    # Clear the songs list
    playlist.delete(*playlist.get_children())
    # Add songs from the selected playlist
    songs = os.listdir(selected_playlist)
    for i, song in enumerate(songs, start=1):
        filename, extension = os.path.splitext(song)
        if extension == '.mp3':
            try:
                audiofile = eyed3.load(os.path.join(selected_playlist, song))
                artist = audiofile.tag.artist
                album = audiofile.tag.album
                add_song_to_dict(filename, i)
                playlist.insert('', 'end', values=(i, filename, artist, album))
            except Exception as e:
                print(f"Error loading file {song}: {e}")


# Function to add a playlist
def add_playlist():
    # Open a directory chooser dialog
    directory = filedialog.askdirectory()
    # Check if the selected directory is already in the playlists list
    if directory not in playlists:
        # Add the selected directory to the playlists list
        playlists.append(directory)
        # Update the playlists list box
        update_playlists()
    else:
        print("This directory is already in the playlists.")


def search_playlist(playlist_name):
    for i, playlist in enumerate(playlists, start=1):
        if playlist_name.lower() in os.path.basename(playlist).lower():
            return i - 1  # return the index of the playlist
    return None  # return None if no match is found


# Create a button for adding playlists
add_playlist_button = tk.Button(playlist_control_frame, text="Add Playlist", command=add_playlist, bg='#7E84F7')
add_playlist_button.pack(side=tk.LEFT)

# Create a list box for displaying playlists
playlists_listbox = Listbox(root, bg='white')
playlists_listbox.place(x=300, y=60, width=590, height=80)  # Adjusted height
playlists_listbox.bind('<<ListboxSelect>>', update_songs)

# Add songs to playlist
songs_dir = 'songs'  # replace with your songs directory
songs = os.listdir(songs_dir)
for i, song in enumerate(songs, start=0):
    filename, extension = os.path.splitext(song)
    if extension == '.mp3':
        try:
            audiofile = eyed3.load(os.path.join(songs_dir, song))
            artist = audiofile.tag.artist
            album = audiofile.tag.album
            add_song_to_dict(filename, i)
            playlist.insert('', 'end', values=(i, filename, artist, album))
        except Exception as e:
            print(f"Error loading file {song}: {e}")

# Define player control functions
is_paused = False
current_song = None


def save_playlists():
    with open('playlists.pkl', 'wb') as f:
        pickle.dump(playlists, f)


def load_playlists():
    try:
        with open('playlists.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


# Function to delete a playlist
def delete_playlist():
    # Check if a playlist is selected
    if playlists_listbox.curselection():
        # Get the selected playlist
        selected_playlist = playlists[playlists_listbox.curselection()[0]]
        # Remove the selected playlist from the playlists list
        playlists.remove(selected_playlist)
        # Update the playlists list box
        update_playlists()
    else:
        print("No playlist selected.")


# Create a button for deleting playlists
delete_playlist_button = tk.Button(playlist_control_frame, text="Delete Playlist", command=delete_playlist,
                                   bg='#7E84F7')
delete_playlist_button.pack(side=tk.LEFT)


# Function to update the playlists list box
def update_playlists():
    # Clear the list box
    playlists_listbox.delete(0, tk.END)
    # Add each playlist to the list box
    for i, playlist in enumerate(playlists, start=1):
        playlists_listbox.insert(tk.END, f"{i}. {os.path.basename(playlist)}")


# Load playlists at the start of the program
playlists = load_playlists()

# Update the playlists list box
update_playlists()


# Function to update the playlists list box
def update_playlists():
    # Clear the list box
    playlists_listbox.delete(0, tk.END)
    # Add each playlist to the list box
    for playlist in playlists:
        playlists_listbox.insert(tk.END, os.path.basename(playlist))


def add_song(filename, i, is_favorite=False):
    # Assuming songs are in the selected playlist directory
    selected_playlist = playlists[playlists_listbox.curselection()[0]]
    song_path = os.path.join(selected_playlist, filename + '.mp3')
    if os.path.exists(song_path):
        try:
            audiofile = eyed3.load(song_path)
            artist = audiofile.tag.artist
            album = audiofile.tag.album
            playlist.insert('', 'end', values=(i, filename, artist, album))
            if is_favorite:
                favorites.append(filename)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
    else:
        print(f"File not found: {song_path}")


def play_song():
    try:
        current_selection = playlist.selection()
        if current_selection:  # if a song is selected
            song = playlist.item(current_selection[0])['values'][1]
        else:  # if no song is selected, default to the first song
            song = playlist.item(playlist.get_children()[0])['values'][1]
    except Exception as e:
        print(f"Error selecting song: {e}")
        return
    # Get the selected playlist
    selected_playlist = playlists[playlists_listbox.curselection()[0]]
    song_path = os.path.join(selected_playlist, song + '.mp3')
    if os.path.exists(song_path):
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        status_label.config(text="Status: Playing")
    else:
        print(f"File not found: {song_path}")


def play_by_index(index):
    global current_song, songs
    # Get the selected playlist
    selected_playlist = playlists[playlists_listbox.curselection()[0]]
    song = playlist.item(playlist.get_children()[index])['values'][1]
    song_path = os.path.join(selected_playlist, song + '.mp3')
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    status_label.config(text="Status: Playing")
    current_song = song
    songs = os.listdir(selected_playlist)


def next_song():
    global current_song
    current_selection = playlist.selection()
    status_label.config(text="Status: Playing")
    if current_selection:  # if a song is selected
        current_index = playlist.get_children().index(current_selection[0])
        next_index = current_index + 1 if current_index + 1 < len(playlist.get_children()) else 0
    else:  # if no song is selected, default to the next song of the currently playing song
        if current_song:  # check if current_song is not False
            current_index = songs.index(current_song)
            next_index = current_index + 1 if current_index + 1 < len(songs) else 0
        else:  # if current_song is False, default to the first song
            next_index = 0
    try:
        playlist.selection_set(playlist.get_children()[next_index])
    except Exception as e:
        print(f"Error selecting next song: {e}")
    play_song()


def previous_song():
    current_selection = playlist.selection()
    status_label.config(text="Status: Playing")
    if current_selection:  # if a song is selected
        current_index = playlist.get_children().index(current_selection[0])
    else:  # if no song is selected, default to the first song
        current_index = 0
    prev_index = current_index - 1 if current_index > 0 else len(playlist.get_children()) - 1
    try:
        playlist.selection_set(playlist.get_children()[prev_index])
    except Exception as e:
        print(f"Error selecting previous song: {e}")
    play_song()


def pause_song():
    global is_paused  # Add this line to access the global variable
    pygame.mixer.music.pause()
    status_label.config(text="Status: Paused")
    is_paused = True  # Update the is_paused status


def stop_song():
    pygame.mixer.music.stop()
    status_label.config(text="Status: Stopped")


def resume_song():
    pygame.mixer.music.unpause()
    status_label.config(text="Status: Resume")
    global is_paused, current_song


# Define volume control function

# Function to update the volume level
current_volume = 40  # Starting volume at 40%


def set_volume(change):
    global current_volume
    current_volume += change
    if current_volume > 100:
        current_volume = 100  # Limit volume to 100%
    elif current_volume < 0:
        current_volume = 0  # Limit volume to 0%
    pygame.mixer.music.set_volume(current_volume / 100)  # Apply volume change to mixer
    print(f"Volume set to: {current_volume}%")  # Replace with actual volume control logic


# previous button
img = Image.open('icons/back.png')
global previous_icon
img = img.resize((40, 40), Image.LANCZOS)
previous_icon = ImageTk.PhotoImage(img)
prev_button = tk.Button(control_frame, image=previous_icon, command=previous_song, bg='#ffffff', compound=tk.CENTER)
prev_button.pack(side=tk.LEFT)
prev_button.pack(side=tk.LEFT, padx=5, pady=5)

# play button
img = Image.open('icons/play.png')
img = img.resize((40, 40), Image.LANCZOS)
play_icon = ImageTk.PhotoImage(img)
play_button = tk.Button(control_frame, image=play_icon, command=play_song, bg='#ffffff', compound=tk.CENTER)
play_button.pack(side=tk.LEFT)
play_button.pack(side=tk.LEFT, padx=5, pady=5)

# pause button
img = Image.open('icons/pause.png')
img = img.resize((40, 40), Image.LANCZOS)
pause_icon = ImageTk.PhotoImage(img)
pause_button = tk.Button(control_frame, image=pause_icon, command=pause_song, bg='#ffffff', compound=tk.CENTER)
pause_button.pack(side=tk.LEFT)
pause_button.pack(side=tk.LEFT, padx=5, pady=5)

# resume button
img = Image.open('icons/resume.png')
img = img.resize((40, 40), Image.LANCZOS)
resume_icon = ImageTk.PhotoImage(img)
resume_button = tk.Button(control_frame, image=resume_icon, command=resume_song, bg='#ffffff', compound=tk.CENTER)
resume_button.pack(side=tk.LEFT)
resume_button.pack(side=tk.LEFT, padx=5, pady=5)

# stop button
img = Image.open('icons/stop.png')
img = img.resize((40, 40), Image.LANCZOS)
stop_icon = ImageTk.PhotoImage(img)
stop_button = tk.Button(control_frame, image=stop_icon, command=stop_song, bg='#ffffff', compound=tk.CENTER)
stop_button.pack(side=tk.LEFT)
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

# next button
img = Image.open('icons/forward.png')
img = img.resize((40, 40), Image.LANCZOS)
next_icon = ImageTk.PhotoImage(img)
next_button = tk.Button(control_frame, image=next_icon, command=next_song, bg='#ffffff', compound=tk.CENTER)
next_button.pack(side=tk.LEFT)
next_button.pack(side=tk.LEFT, padx=5, pady=5)

# Volume up button
img = Image.open('icons/volume_up.png')
img = img.resize((40, 40), Image.LANCZOS)
volume_up_icon = ImageTk.PhotoImage(img)
volume_up_button = tk.Button(control_frame, image=volume_up_icon, command=lambda: set_volume(10), bg='#ffffff',
                             compound=tk.CENTER)
volume_up_button.pack(side=tk.LEFT, padx=5, pady=5)

# Volume down button
img = Image.open('icons/volume-down.png')
img = img.resize((40, 40), Image.LANCZOS)
volume_down_icon = ImageTk.PhotoImage(img)
volume_down_button = tk.Button(control_frame, image=volume_down_icon, command=lambda: set_volume(-10), bg='#ffffff',
                               compound=tk.CENTER)
volume_down_button.pack(side=tk.LEFT, padx=5, pady=5)

# Function to update the volume level
current_volume = 40  # Starting volume at 40%


def set_volume(change):
    global current_volume
    change = int(change)  # Convert change to integer
    current_volume += change
    if current_volume > 100:
        current_volume = 100  # Limit volume to 100%
    elif current_volume < 0:
        current_volume = 0  # Limit volume to 0%
    pygame.mixer.music.set_volume(current_volume / 100)  # Apply volume change to mixer
    print(f"Volume set to: {current_volume}%")  # Replace with actual volume control logic


def on_exit():
    # Save playlists when the application is closed
    save_playlists()
    pygame.mixer.music.stop()
    root.quit()

    # Bind the exit event to the function
    root.protocol("WM_DELETE_WINDOW", on_exit)

    # Start the Tkinter event loop
    root.mainloop()

    # Stop the song if it's playing
    pygame.mixer.music.stop()

    # Stop the voice command listening thread
    if voice_thread.is_alive():
        # Set a flag that will cause the thread to exit
        global exit_flag
        exit_flag = True

    # Destroy the tkinter window
    root.destroy()

    # Force terminate the Python script
    os._exit(0)


# Set the flag to False initially
exit_flag = False

# Bind the exit function to the window close button
root.protocol("WM_DELETE_WINDOW", on_exit)


def handle_voice_command(recognizer, microphone):
    global current_song, songs
    while not exit_flag:
        with microphone as source:
            print("Listening for command...")
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
                command = recognizer.recognize_google(audio)
                command = command.lower()
                print("You said: " + command)

                # Update voice command label
                voice_command_label.config(text=f"Voice Command: {command}")

                # Command handling
                if command in play_commands:
                    play_song()
                elif command in stop_commands:
                    stop_song()
                elif command in next_song_commands:
                    next_song()
                elif command in previous_song_commands:
                    previous_song()
                elif command in pause_commands:
                    pause_song()
                elif command in volume_up_commands:
                    set_volume(10)
                elif command in volume_down_commands:
                    set_volume(-10)
                elif any(cmd in command for cmd in search_playlist_commands):
                    search_term = command.split(" ", 1)[1]
                    playlist_index = search_playlist(search_term)
                    if playlist_index is not None:
                        playlists_listbox.selection_clear(0, tk.END)
                        playlists_listbox.selection_set(playlist_index)
                        playlists_listbox.event_generate("<<ListboxSelect>>")
                elif command.split(" ")[0] == "play" and command.split(" ")[1] == "song" and len(
                        command.split(" ")) > 2:
                    search_term = " ".join(command.split(" ")[2:])
                    song_index = search_song(search_term)
                    if song_index is not None:
                        play_by_index(song_index)
                else:
                    print("Command not recognized")

            except sr.UnknownValueError:
                print("Sorry, I did not understand the audio")

            except WaitTimeoutError:
                print("Timeout waiting for audio")


favorites = []


def toggle_favorite():
    selected_song = playlist.item(playlist.selection())['values'][1]
    selected_playlist = playlists[playlists_listbox.curselection()[0]]
    song_path = os.path.join(selected_playlist, selected_song + '.mp3')
    if song_path in favorites:
        favorites.remove(song_path)
        print(f"Removed {selected_song} from favorites")
    else:
        favorites.append(song_path)
        print(f"Added {selected_song} to favorites")


def show_favorites():
    # Clear the playlist display
    playlist.delete(*playlist.get_children())
    # Add favorite songs to the playlist display
    for i, song_path in enumerate(favorites, start=1):
        try:
            audiofile = eyed3.load(song_path)
            artist = audiofile.tag.artist
            album = audiofile.tag.album
            filename = os.path.splitext(os.path.basename(song_path))[0]
            playlist.insert('', 'end', values=(i, filename, artist, album))
        except Exception as e:
            print(f"Error loading file {song_path}: {e}")

# favourite button
# img = Image.open('icons/favorite.png')
# img = img.resize((50, 50), Image.LANCZOS)
# favourite_icon = ImageTk.PhotoImage(img)
# favourite_button = tk.Button(control_frame, image=favourite_icon, command=show_favorite_songs, bg='#ffffff',
#                              compound=tk.CENTER)
# favourite_button.pack(side=tk.LEFT)
# favourite_button.pack(side=tk.LEFT, padx=5, pady=5)

# Define volume_scale
volume_scale = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Volume", command=set_volume,
                        bg='#ffffff')
volume_scale.set(40)  # Set default volume to 40%
volume_scale.pack(side=tk.RIGHT)

gesture_list = []
gesture_counter = 0


def search_song_visual():
    global gesture_counter
    gesture_counter = 0  # Reset the counter when 'S' is detected
    print("Search mode activated. Waiting for two word gestures...")


def play_search_song():
    global gesture_counter
    if gesture_counter >= 2:
        search_term = ' '.join(set(gesture_list[-2:]))  # Use set to ensure unique words
        song_index = search_song(search_term)
        if song_index is not None:
            print("Playing song ", song_index, ": ", search_term)
            play_by_index(song_index)
            current_song = search_term  # Update the current_song variable
            songs = [playlist.item(item)['values'][1] for item in playlist.get_children()]  # Update the songs list
            playlist.selection_set(playlist.get_children()[song_index])  # Select the song in the playlist
        else:
            print("Song not found: " + search_term)
        gesture_list.clear()  # Clear the gesture list after search
        gesture_counter = 0  # Reset the counter
    else:
        print("Not enough gestures for search")


def gesture_control(character):
    global gesture_counter
    if character == 'Play':
        play_song()
    elif character == 'Stop':
        stop_song()
    elif character == 'Next':
        next_song()
    elif character == 'Back':
        previous_song()
    elif character == 'VolumeUp':
        set_volume(10)
    elif character == 'VolumeDown':
        set_volume(-10)
    elif character == 'Search':
        search_song_visual()
    elif character == 'H':
        gesture_list.append('be')
        gesture_counter += 1
        play_search_song()
    elif character == 'Y':
        gesture_list.append('more')
        gesture_counter += 1
        play_search_song()
    elif character == 'G':
        gesture_list.append('bones')
        gesture_counter += 1
        play_search_song()


def recognize_hand_gestures():
    model_dict = pickle.load(open('gesture/model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {
        0: 'Play', 1: 'Stop', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'VolumeUp', 11: 'Back', 12: 'VolumeDown', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'Search',
        19: 'Next', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }

    max_length = 42

    while not exit_flag:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            try:
                # Pad data_aux to the required length
                data_aux = np.pad(data_aux, (0, max_length - len(data_aux)), 'constant')
            except ValueError as e:
                print(f"Error padding data: {e}")
                continue

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            print("Predicted character: ", predicted_character)
            gesture_control(predicted_character)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Global variable to keep track of the current mode
current_mode = "voice"  # Default mode is voice


# Function to toggle between voice and gesture modes
def toggle_mode():
    global current_mode, voice_thread, gesture_thread, exit_flag

    if current_mode == "voice":
        current_mode = "gesture"
        print("Switched to gesture mode")
        mode_toggle_button.config(text="Switch to Voice Mode")

        # Stop the voice command thread
        exit_flag = True
        if voice_thread.is_alive():
            voice_thread.join()

        # Start the gesture command thread
        exit_flag = False
        gesture_thread = threading.Thread(target=recognize_hand_gestures)
        gesture_thread.start()

    else:
        current_mode = "voice"
        print("Switched to voice mode")
        mode_toggle_button.config(text="Switch to Gesture Mode")

        # Stop the gesture command thread
        exit_flag = True
        if gesture_thread.is_alive():
            gesture_thread.join()

        # Start the voice command thread
        exit_flag = False
        voice_thread = threading.Thread(target=handle_voice_command, args=(recognizer, microphone))
        voice_thread.start()

img = Image.open('icons/V_G.png')
img = img.resize((40, 40), Image.LANCZOS)
hand = ImageTk.PhotoImage(img)

# Initialize speech recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Add a button to toggle between voice and gesture modes
mode_toggle_button = tk.Button(control_frame, text="Switch to Gesture Mode", command=toggle_mode, image=hand)
mode_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)

def show_favorite_songs():
    show_favorites()


# Create a button to toggle favorite status
toggle_favorite_button = tk.Button(control_frame, text="Toggle Favorite", command=toggle_favorite, bg='#7E84F7')
show_favorites_button = tk.Button(control_frame, text="Show Favorites", command=show_favorites, bg='#7E84F7')
toggle_favorite_button.pack(side=tk.LEFT, padx=5, pady=5)

# Create a button to show favorite songs
show_favorites_button = tk.Button(control_frame, text="Show Favorites", command=show_favorites, bg='#7E84F7')
show_favorites_button.pack(side=tk.LEFT, padx=5, pady=5)


# Start the voice command handler in a separate thread
exit_flag = False
voice_thread = threading.Thread(target=handle_voice_command, args=(recognizer, microphone))
voice_thread.start()

# Run the main loop
root.mainloop()

# add a seperator toggle for gesture and voice control
# implement favourite
# search song with gesture
# add camera video to the player frame
