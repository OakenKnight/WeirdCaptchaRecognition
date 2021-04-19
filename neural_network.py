import numpy as bb8
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD


def train_model(x_train, y_train, serialization_folder):
    trained_model = None
    neural_network = create_network()

    trained_model = train_network(neural_network, x_train, y_train)

    serialize_ann(trained_model, serialization_folder)

    return trained_model


def serialize_ann(nn, serialization_folder):
    print("Saving network...")
    model_json = nn.to_json()
    with open(serialization_folder + "/neuronska.json", "w") as json_file:
        json_file.write(model_json)

    nn.save_weights(serialization_folder + "/neuronska.h5")

    print("Network saved successfully!")


def train_network(neural_network, x_train, y_train):
    print("Training network...")

    x_train = bb8.array(x_train, bb8.float32)
    y_train = bb8.array(y_train, bb8.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    neural_network.compile(loss='categorical_crossentropy', optimizer=sgd)

    neural_network.fit(x_train, y_train, epochs=6000, batch_size=1, verbose=1, shuffle=False)
    print("Network trained successfully!")
    return neural_network


def create_network():
    print("Creating network...")

    neural_network = Sequential()
    neural_network.add(Dense(128, input_dim=784, activation='sigmoid'))
    neural_network.add(Dense(60, activation='sigmoid'))

    print("Network created successfully!")
    return neural_network


def load_trained_model(serialization_folder):
    try:
        print("Loading trained model....")
        json_file = open(serialization_folder + "/neuronska.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        network = model_from_json(loaded_model_json)

        network.load_weights(serialization_folder + "/neuronska.h5")
        print("Trained model found successfully!")
        return network
    except Exception as e:
        print("Warning: No model found!")
        return None
