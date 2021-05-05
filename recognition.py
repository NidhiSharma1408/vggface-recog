from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
import time
import pickle


detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def find_face(pixels):
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1,y1,x2,y2=0,0,0,0
    found = False
    for result in results:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        found = True
        # extract the face    
    return found,x1,y1,x2,y2


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    found,x1,y1,x2,y2 = find_face(pixels)
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def get_embedding(faces):
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    return yhat


def get_face_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    return get_embedding(faces)


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False
 

def recognise(embeddings,names,required_size=(224, 224)):
    cap = cv2.VideoCapture(0)
    face_array = []
    start = time.time()
    predictions = []
    while len(predictions)<10:
        ok, frame = cap.read()
        if not ok:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_array = []
        s = time.time()
        f,x1,y1,x2,y2=find_face(small_frame)
        print(time.time()-s,"in face find")
        if not f:
            continue
        found = False
        face = small_frame[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))
        s = time.time()
        embedding = get_embedding(face_array)[0]
        img = cv2.rectangle(frame, (x1*4, y1*4), (x2*4, y2*4), (0, 255, 0), 2)
        print(time.time()-s," in embedding")
        i=0
        for em in embeddings:
            if is_match(embedding,em):
                print(names[i])
                found = True
                predictions.append(names[i])
                img = cv2.putText(frame, names[i], (x1*4,y1*4), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0),2, cv2.LINE_AA)
                break
            i+=1
        if not found:
            img = cv2.putText(frame, 'none', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0),2, cv2.LINE_AA)
            predictions.append('none')
            print('not in database')

        cv2.imshow('face recog', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return max(set(predictions), key = predictions.count)

def capture_face():
    required_size=(224, 224)
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ok = not ok
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        found, x1, y1,x2,y2 = find_face(small_frame)

        if not found:
            img = cv2.putText(frame, 'face not visible', (10,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2, cv2.LINE_AA)
        else:
            img = cv2.putText(frame, 'Press C when ready', (10,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2, cv2.LINE_AA)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                face = small_frame[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face = asarray(image)
                embedding = get_embedding([face])[0]
                break
        cv2.imshow('new face',img)
    cap.release()
    cv2.destroyAllWindows()
    return ok,embedding
# filenames = ['images/sapna.jpg','images/himanshi.jpg','images/nidhi.jpg','images/mummy.jpg','images/kalam.jpg','images/srk.jpg','images/messi.jpg']
# # get embeddings file filenames
# embeddings = get_face_embeddings(filenames)

# all_face_encodings = {}
# for i in range(len(filenames)):
#     all_face_encodings[filenames[i][7:-4]] = embeddings[i]

# with open('dataset_faces.dat', 'wb') as f:
#     pickle.dump(all_face_encodings, f)






