import tkinter as tk
import os
from recognition import *

# load existing embeddings
with open('dataset_faces.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)
names = list(all_face_encodings.keys())
embeddings = list(all_face_encodings.values())


root = tk.Tk()
root.title("Face Recognition")

def captureFace(root2, name):
    print(name)
    ok,embedding = capture_face()
    if ok:
        all_face_encodings[name] = embedding
        with open('dataset_faces.dat', 'wb') as f:
            pickle.dump(all_face_encodings, f)
        names.append(name)
        embeddings.append(embedding)
    root2.destroy()
    return


def addFace():
    root2 = tk.Tk()
    root2.title("Add Face")
    name = tk.Entry(root2, width=30)
    name.grid(row=0,column=0)
    tk.Button(root2, text="submit", command=lambda: captureFace(root2, name.get())).grid(row=1, column=0)
    root2.mainloop()
    return


def attendence():
    print("taking attendence")
    name = recognise(embeddings, names)
    tk.Label(root, text=f"You are {name}").grid(row=4,column=2)
    return


tk.Label(root, text="Face Attencdence System").grid(row=0, column=0)
tk.Button(root, text="Add face", padx=10, pady=10, command=addFace).grid(row=1, column=0)
tk.Button(root, text="Mark Attendence", padx=10, pady=10, command=attendence).grid(row=2, column=0)

# entry = tk.Entry(root, width=10, bg='Green', fg='White', borderwidth=5)
# entry.grid(row=3,column=2)
# entry.insert(0, "Enter your name!")

root.mainloop()