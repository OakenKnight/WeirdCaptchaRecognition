# import libraries here
import numpy as bb8

import handle_data as hd
import handle_image as hi
import neural_network as nn
import cv2

def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    prepared_letters = hi.get_letters_train(train_image_paths[0])

    x_train = hd.prepare_data_for_network(prepared_letters)
    y_train = hd.convert_output()

    model = nn.load_trained_model(serialization_folder)

    if model is None:
        model = nn.train_model(x_train, y_train, serialization_folder)

    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    letters, k_means = hi.get_letters_k_means_test(image_path)
    inputs = hd.prepare_data_for_network(letters)
    result = trained_model.predict(bb8.array(inputs, bb8.float32))
    if k_means is not None:
        text = hd.display_result(result, k_means)
    else:
        text="Error in ??egmentation"
    extracted = hd.do_fuzzywuzzy_stuff(vocabulary,text)
    print(extracted)
    return extracted

