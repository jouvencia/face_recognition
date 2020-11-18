#SETUP
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import operator

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


#Functions
def facetensordatabase(train_dir, verbose=False):
    """
    
    Transform a database pictures of faces into a encoding faces database.
    
    :param train_dir: path to the directory that contains a sub-directory for each known person, with its name.
    10 face pictures/ person should be good (different orientation, but not fuzzy).
     Structure for the database:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
        
    :param verbose: verbosity of training
    :return: returns 2 variables: * X: list of all the tensors (encoding face, shape of 128 each). lenght of X should be nbr
                                    of person in the database* nbr of faces/person 
                                    (if 10 faces/ person, lenght = 10 * nbr of person)
                                  * y: list of the name corresponding to this encoding face 
                                       (name of the folder containing the faces)
                                       lenght of y= lenght of X
                                       
    """
    #loading  the facetensorDatabase to update:
    with open('facedatabase', 'rb') as f:
    X = pickle.load(f)
    
    with open('Name','rb') as f:
    Y = pickle.load(f)
    
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    return X, y


def euclideanDistance(instance1, instance2, length):
    """
    
    Calculate a euclidean distance between 2 tensors. 
    The lenght correspond to the lenght of the tensors after encoding a face it will always be 128.
    
    :param instance1: encoding face tensor (shape 128)
    :param instance1: encoding face tensor to compare (shape 128)
    :param lenght: lenght of the tensor. Here it will be always 128 (output of face_recognition.face_encodings format)
    
    :return: returns 1 variables: A float cooresponding to the euclidean Distance between two tensors.
    
    """
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getKNeighbors(X, y, image_to_test, k):
    """
    
    This function calculate the distance between an encoding face test_image and the encoding face database. 
    Then it select the k closest encoding face. 
    
    :param X: list of all the encoding face database (output of facetensordatabase() function)
    :param y: list of the name corresponding to these encoding face (output of facetensordatabase() function)
    :param image_to_test: encoding face to test against the database
    :param k: numbers of closest neighbours we want to use for the KNN
    
    :return: returns 1 variables: A list containing tuple organised as follow: (face_encoding tensor, distance, name)
    
    """
    distances = []
    length = len(image_to_test)
    for x in range(len(facedatabase)):
        dist = euclideanDistance(image_to_test, X[x], length)
        distances.append((X[x], dist, y[x]))
    distances.sort(key=operator.itemgetter(1))
    close_neighbors = []
    for x in range(k):
        close_neighbors.append((distances[x][0],distances[x][1], distances[x][2]))
    return close_neighbors


def getResponse(X, distance_threshold):
    """
    
    This function is used to classify the unknown encoding face to a corresponding member of the database when the distance is 
    close enough.
    
    :param X: output of getKNeighbors() function
    :param threshold: if the distance between the unknown encoding face and the closest neighbour is >0.6 than 
                      the unknown encoding face is classify as unknown
                      
    :return: returns 2 variables: * sorted_Vo: A list containing the name getting the majority of vote.
                                  * sorted_Vo: A list of Bol saying if the is a match or not with database
                                              True=yes
                                              False=No
    
    """
    classVotes = {}
    classVotes["Unknown Person"] = 0
    for x in range(len(X)):
        response = X[x][-1]
        thresh= X[x][-2]
        if thresh <= distance_threshold:
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        else:
            classVotes["Unknown Person"] +=1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    are_matches = [sortedVotes[0][i] != "Unknown Person" for i in range (1)]
    sorted_Vo=[]
    sorted_Vo.append(sortedVotes[0][0])
    return sorted_Vo, are_matches

def predict(X_img_path, distance_threshold=0.6, k=3):
    """
    Recognizes faces in given image using a KNN classifier
    
    :param X_img_path: path to image to be recognized
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :param k: numbers of closest neighbours we want to use for the KNN
    
    
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
        
    """
    
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    nbr_of_faces = len(X_face_locations)

    # If no faces are found in the image, return an empty result.
    if nbr_of_faces == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    
    # Prepare the output image
    pil_image = Image.fromarray(X_img)
    draw = ImageDraw.Draw(pil_image)
       
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(X_face_locations, faces_encodings):
        
    # See if the face is a match for the known face(s)
        neighbors = getKNeighbors(facedatabase, Name, face_encoding, k)
        predi, are_matches = getResponse(neighbors, distance_threshold)
                 
        name = "Unknown"
        if are_matches:
            name = predi[0]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))   
        
    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    return pil_image

#Get the new database update:
Path="./facial_recog_test/train"
facedatabase, Name=facetensordatabase(train_dir=Path, verbose=True)

#save the database
with open('facedatabase', 'wb') as f:
    pickle.dump(facedatabase, f)
    
with open('Name','wb') as f:
    pickle.dump(Name, f)

#to figure out how many encoding faces£/person are present in the database:
name_dictionnary={Names: len(Names) for Names in Name}
print("there are: ", len(name_dictionnary), "persons in the database")
name_dictionnary


#to make a prediction:
#for image_file in os.listdir("./test"):
#    full_file_path = os.path.join("./test", image_file)
#
#    print("Looking for faces in {}".format(image_file))
#
#    display(predict(full_file_path, distance_threshold=0.45, k=5))
