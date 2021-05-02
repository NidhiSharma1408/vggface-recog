from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
import time
def find_face(pixels):
    detector = MTCNN()
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
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
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
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        return True
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        return False
 
def recognise(embeddings,filenames,required_size=(224, 224)):
    
    cap = cv2.VideoCapture(0)
    face_array = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        face_array = []
        f,x1,y1,x2,y2=find_face(frame)
        if not f:
            continue
        found = False
        face = frame[y1:y2, x1:x2]
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))
        embedding = get_embedding(face_array)[0]
        i=0
        
        for em in embeddings:
            if is_match(embedding,em):
                
                print(filenames[i][8:-4])
                found = True
                img = cv2.putText(frame, filenames[i][7:-4], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0),2, cv2.LINE_AA)
                break
            i+=1
            
        if not found:
            
            img = cv2.putText(frame, 'none', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0),2, cv2.LINE_AA)
        time.sleep(3)
        cv2.imshow('face recog', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

filenames = ['images/sapna.jpg','images/himanshi.jpg','images/nidhi.jpg','images/mummy.jpg','images/kalam.jpg','images/srk.jpg','images/messi.jpg']
# get embeddings file filenames
embeddings = get_face_embeddings(filenames)
recognise(embeddings,filenames)



