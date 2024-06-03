import os
import face_recognition
import numpy as np

def encode_faces(directory):
    known_encodings = []
    known_names = []
    
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
    
    return known_encodings, known_names

known_encodings, known_names = encode_faces('path_to_images_directory')

np.save('encodings.npy', known_encodings)
np.save('names.npy', known_names)
