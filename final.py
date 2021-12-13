# read line by line and if the line does not contain a '+' then remove it and return an array of strings that represent one image
from io import StringIO
import math
from os import remove
import random
from abc import abstractproperty
from types import coroutine
from typing import final
import numpy as np
import time


class Read():
    def read_digits(file_path):
        image = []
        images = []
        counter = 0
        image_string = ""
        file = open(file_path, 'r')
        for line in file:
            count = line.count(' ') + line.count('\n')
            if count == len(line) and len(image_string) > 0:
                image.append(image_string)
                image_string = ""
            elif line.count('+') > 0 or line.count('#') > 0:
                image_string += line
        for i in range(len(image)):
            if image[i].count('\n') <= 5:
                continue
            else:
                images.append(image[i])
        del image
        return images

    def read_faces(file_path):
        image = []
        image_string = ""
        file = open(file_path, 'r')
        i = 1
        for line in file:
            if i % 70 == 0:
                image.append(image_string)
                image_string = ""
            image_string += line
            i += 1

        return image

    def read_labels(file_path):
        try:
            digits = []
            file = open(file_path, 'r')
            for line in file:
                digits.append(line.rstrip())
            return digits
        except IOError:
            print("incorrect file_path provided for training labels")


class Features():
    def percentage_filled(string):
        full = 0
        total = 1
        for i in range(len(string)):
            if string[i] == '+':
                full += .5
            if string[i] == '#':
                full += 1
            total += 1
        percentage = full/total
        return percentage

    def string_to_matrix(image):
        matrix = []
        split_string = image.split('\n')
        for i in range(len(split_string)):
            matrix.append([])
            matrix[i] = list(split_string[i])
        return matrix


    def string_to_matrix2(image):
        matrix = np.zeros((70, 60))
        split_string = image.split('\n')
        
        for str in split_string:
            i = 0
            for j in range(len(str)):
                matrix[i][j] = 0 if str[j] == '#' else 1
            i += 1
        return matrix

    def quadrant_one(image):
        mat = Features.string_to_matrix(image)
        quadrant_one = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])//2):
                quadrant_one = quadrant_one+mat[i][j]
        return Features.percentage_filled(quadrant_one)

    def quadrant_two(image):
        mat = Features.string_to_matrix(image)
        quadrant_two = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])//2, len(mat[0])):
                quadrant_two = quadrant_two+mat[i][j]
        return Features.percentage_filled(quadrant_two)

    def quadrant_three(image):
        mat = Features.string_to_matrix(image)
        quadrant_three = ""
        for i in range(len(mat)//2, len(mat)-1):
            for j in range(len(mat[0])//2):
                quadrant_three = quadrant_three+mat[i][j]
        return Features.percentage_filled(quadrant_three)

    def quadrant_four(image):
        mat = Features.string_to_matrix(image)
        quadrant_four = ""
        for i in range(len(mat)//2, len(mat)-1):
            for j in range(len(mat[0])//2, len(mat[0])):
                quadrant_four = quadrant_four+mat[i][j]
        return Features.percentage_filled(quadrant_four)

    def top_heavy(image):
        mat = Features.string_to_matrix(image)
        top_half = ""
        for i in range(len(mat)//2):
            for j in range(len(mat[0])):
                top_half += mat[i][j]
        retval = 1 if Features.percentage_filled(top_half) > .7 else 0
        return retval

    def get_features(image):
        mat = Features.string_to_matrix(image)
        feat = [1]
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)
        q1 = Features.quadrant_one(image)
        q2 = Features.quadrant_one(image)
        q3 = Features.quadrant_one(image)
        q4 = Features.quadrant_one(image)
        top_hvy = Features.top_heavy(image)
        feat.append(q1)
        feat.append(q2)
        feat.append(q3)
        feat.append(q4)
        feat.append(top_hvy)
        return feat

    def get_features_nb(image):
        mat = Features.string_to_matrix(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)

        top_hvy = Features.top_heavy(image)
        feat.append(top_hvy)

        return feat

    def get_face_features(image):
        mat = Features.string_to_matrix(image)
        feat = [1]
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#':
                    feat.append(1)
                else:
                    feat.append(0)
        return feat

    def get_face_features_nb(image):
        mat = Features.string_to_matrix(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#':
                    feat.append(1)
                else:
                    feat.append(0)
        return feat

    def get_features_knn(image):
        mat = Features.string_to_matrix(image)
        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#' or mat[i][j] == '+':
                    feat.append(1)
                else:
                    feat.append(0)
        while (len(feat) < 588):
            feat.append(0)
        return feat

    def get_face_features_knn(image):
        mat = Features.string_to_matrix(image)



        feat = []
        for i in range(len(mat)-1):
            for j in range(len(mat[0])):
                if mat[i][j] == '#':
                    feat.append(1)
                    # print(mat[i][j],end="")
                else:
                    feat.append(0)
                    # print(mat[i][j],end="")
        #     print()

        # print(f"{len(mat)}     {len(mat[0])}")
        while (len(feat) < 4200):
            feat.append(0)
        return feat

    def get_face_features_knn2(image):
        mat = Features.string_to_matrix2(image)
        split_test = Features.split(mat, 14, 12)
        perc_fill = []

        for i in range(len(split_test)):
            count = 0
            for j in split_test[i]:
                count += j
                if j != 0 :
                    count += 1
            perc_fill.append(count / 182.0)

        return perc_fill
        # for i in range(25):
        #     print(perc_fill[i])   



    def split(array, nrows, ncols):
        """Split a matrix into sub-matrices."""
        sub_mat = [[] for i in range(25)]
        for i in range(len(array)):
            for j in range(len(array[0])):
                # print(f"{i} {j}   {int(i / 5)+int(j / 5)}")
                sub_mat[int(i / 5)+int(j / 5)].append(array[i][j])

        return sub_mat

class Perceptron():
    def initialize_weights(w_map, digits, num_features):
        for i in range(digits):
            w_map[str(i)] = [random.random() for _ in range(num_features)]
            # w_map[str(i)].append(1)
        return w_map

    def weight_adjustments(multi_classes, predicted_value, real_value, features):
        if predicted_value != real_value:
            for i in range(len(features)):
                multi_classes[predicted_value][i] = multi_classes[predicted_value][i]-features[i]
            for i in range(len(features)):
                multi_classes[real_value][i] = multi_classes[real_value][i]+features[i]

    def face_weight_adjustments(weights_list, predicted_value, real_value, features):
        if predicted_value > 0 and real_value == '0':
            for i in range(len(weights_list)):
                weights_list[i] = weights_list[i]-features[i]
        elif predicted_value < 0 and real_value == '1':
            for i in range(len(weights_list)):
                weights_list[i] = weights_list[i]+features[i]


def train_perceptron(data_set, train_amount=450):
    if data_set == 'digits':
        images = Read.read_digits("digitdata/trainingimages")
        lables = Read.read_labels("digitdata/traininglabels")
        digit_weights = {'0': [], '1': [], '2': [], '3': [],
                         '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
        len_features = len(Features.get_features(images[0]))
        digit_weights = Perceptron.initialize_weights(
            digit_weights, 10, len_features)
        function_map = {'0': 0, '1': 0, '2': 0, '3': 0,
                        '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for i in range(train_amount):
            features = Features.get_features(images[i])
            for digit, weights in digit_weights.items():
                func_val = 0
                for w in range(len(features)):
                    func_val += features[w]*weights[w]
                function_map[digit] = func_val
            prediction = max(function_map, key=function_map.get)
            Perceptron.weight_adjustments(
                digit_weights, prediction, lables[i], features)
        return digit_weights
    else:
        images = Read.read_faces('facedata/facedatatrain')
        lables = Read.read_labels('facedata/facedatatrainlabels')
        len_features = len(Features.get_face_features(images[0]))
        face_weights = [random.random() for _ in range(len_features)]
        for i in range(train_amount):
            func_val = 0
            features = Features.get_face_features(images[i])
            for w in range(len(face_weights)):
                func_val += face_weights[w]*features[w]
            Perceptron.face_weight_adjustments(
                face_weights, func_val, lables[i], features)
        return face_weights
def test_perceptron(data_set, testamount, train_amount):
    weights_dict = train_perceptron(data_set, train_amount)
    if data_set == 'digits':
        images = Read.read_digits(
            'digitdata/testimages')
        lables = Read.read_labels(
            "digitdata/testlabels")
        correct = 0
        function_map = {'0': 0, '1': 0, '2': 0, '3': 0,
                        '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for i in range(testamount):
            features = Features.get_features(images[i])
            for digit, weights in weights_dict.items():
                func_val = 0
                for w in range(len(features)):
                    func_val += features[w]*weights[w]
                func_val += weights[-1]
                function_map[digit] = func_val
            prediction = max(function_map, key=function_map.get)
            if prediction == lables[i]:
                # print('Correct: ', correct, ' Prediction:',
                #       prediction, " Actual:", lables[i])
                correct += 1
        return correct/testamount
    else:
        images = Read.read_faces(
            'facedata/facedatatest')
        lables = Read.read_labels(
            'facedata/facedatatestlabels')
        correct = 0
        for i in range(testamount):
            func_val = 0
            prediction = '0'
            features = Features.get_face_features(images[i])
            for w in range(len(weights_dict)):
                func_val += weights_dict[w]*(features[w])
            if func_val > 0:
                prediction = '1'
            elif func_val <= 0:
                prediction = '0'
            if str(prediction) == lables[i]:
                correct += 1
                # print('Correct: ', correct, ' Prediction:',
                #       prediction, " Actual:", lables[i])
        return correct/testamount


class Bayes():
    def clean_changes(zero, one, two, three, four, five, six, seven, eight, nine, prior_dict):
        for key, val in zero.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['0']
        for key, val in one.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['1']
        for key, val in two.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['2']
        for key, val in three.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['3']
        for key, val in four.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['4']
        for key, val in five.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['5']
        for key, val in six.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['6']
        for key, val in seven.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['7']
        for key, val in eight.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['8']
        for key, val in nine.items():
            for i in range(len(val)):
                if val[i] == 0:
                    val[i] = .01
                else:
                    val[i] = val[i]/prior_dict['9']

    def clean_prediction(p_zero, p_one, p_two, p_three, p_four, p_five, p_six, p_seven, p_eight, p_nine):
        max_p = max([p_zero, p_one, p_two, p_three, p_four,
                    p_five, p_six, p_seven, p_eight, p_nine])
        prediction = ''
        if max_p == p_zero:
            prediction = '0'
        elif max_p == p_one:
            prediction = '1'
        elif max_p == p_two:
            prediction = '2'
        elif max_p == p_three:
            prediction = '3'
        elif max_p == p_four:
            prediction = '4'
        elif max_p == p_five:
            prediction = '5'
        elif max_p == p_six:
            prediction = '6'
        elif max_p == p_seven:
            prediction = '7'
        elif max_p == p_eight:
            prediction = '8'
        elif max_p == p_nine:
            prediction = '9'

        return prediction


def train_bayes_digits(train_amount):
    images = Read.read_digits('digitdata/trainingimages')
    lables = Read.read_labels('digitdata/traininglabels')
    prior_dict = {}
    for i in range(train_amount):
        if lables[i] in prior_dict:
            prior_dict[lables[i]] += 1
        else:
            prior_dict[lables[i]] = 1
    length = len(Features.get_features_nb(images[0]))
    zero = {i: [0, 0] for i in range(length)}
    one = {i: [0, 0] for i in range(length)}
    two = {i: [0, 0] for i in range(length)}
    three = {i: [0, 0] for i in range(length)}
    four = {i: [0, 0] for i in range(length)}
    five = {i: [0, 0] for i in range(length)}
    six = {i: [0, 0] for i in range(length)}
    seven = {i: [0, 0] for i in range(length)}
    eight = {i: [0, 0] for i in range(length)}
    nine = {i: [0, 0] for i in range(length)}
    for i in range(train_amount):
        if lables[i] == '0':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    zero[j][1] += 1
                else:
                    zero[j][0] += 1
        elif lables[i] == '1':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    one[j][1] += 1
                else:
                    one[j][0] += 1
        elif lables[i] == '2':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    two[j][1] += 1
                else:
                    two[j][0] += 1
        elif lables[i] == '3':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    three[j][1] += 1
                else:
                    three[j][0] += 1
        elif lables[i] == '4':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    four[j][1] += 1
                else:
                    four[j][0] += 1
        elif lables[i] == '5':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    five[j][1] += 1
                else:
                    five[j][0] += 1
        elif lables[i] == '6':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    six[j][1] += 1
                else:
                    six[j][0] += 1
        elif lables[i] == '7':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    seven[j][1] += 1
                else:
                    seven[j][0] += 1
        elif lables[i] == '8':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    eight[j][1] += 1
                else:
                    eight[j][0] += 1
        elif lables[i] == '9':
            features = Features.get_features_nb(images[i])
            for j in range(len(features)):
                if features[j] == 1:
                    nine[j][1] += 1
                else:
                    nine[j][0] += 1
    Bayes.clean_changes(zero, one, two, three, four, five,
                        six, seven, eight, nine, prior_dict)
    return prior_dict, train_amount, zero, one, two, three, four, five, six, seven, eight, nine


def train_bayes_faces(train_amount):
    images = Read.read_faces('facedata/facedatatrain')
    lables = Read.read_labels('facedata/facedatatrainlabels')
    prior_dict = {}
    for i in range(train_amount):
        if lables[i] in prior_dict:
            prior_dict[lables[i]] += 1
        else:
            prior_dict[lables[i]] = 1
    face_training_dict = {i: [0, 0] for i in range(60*70)}
    not_face_training_dict = {i: [0, 0] for i in range(60*70)}
    for i in range(train_amount):
        if lables[i] == '1':
            features = Features.get_face_features_nb(images[i])
            for i in range(len(features)):
                if features[i] == 1:
                    face_training_dict[i][1] += 1
                else:
                    face_training_dict[i][0] += 1
        elif lables[i] == '0':
            features = Features.get_face_features_nb(images[i])
            for i in range(len(features)):
                if features[i] == 1:
                    not_face_training_dict[i][1] += 1
                else:
                    not_face_training_dict[i][0] += 1
    for key, val in face_training_dict.items():
        for i in range(len(val)):
            if val[i] == 0:
                val[i] = .01
            else:
                val[i] = val[i]/prior_dict['1']
    for key, val in not_face_training_dict.items():
        for i in range(len(val)):
            if val[i] == 0:
                val[i] = .01
            else:
                val[i] = val[i]/prior_dict['0']
    return prior_dict, train_amount, face_training_dict, not_face_training_dict


def test_bayes_digits(test_amount = 1000, train_amount = 4000):
    prior, n, zero,one,two,three,four,five,six,seven,eight,nine= train_bayes_digits(train_amount)
    images = Read.read_digits('digitdata/testimages')
    labels = Read.read_labels('digitdata/testlabels')
    correct = 0
    for j in range(test_amount):
        p_zero = math.log(prior['0']/n)
        p_one = math.log(prior['1']/n)
        p_two = math.log(prior['2']/n)
        p_three = math.log(prior['3']/n)
        p_four = math.log(prior['4']/n)
        p_five = math.log(prior['5']/n)
        p_six = math.log(prior['6']/n)
        p_seven = math.log(prior['7']/n)
        p_eight = math.log(prior['8']/n)
        p_nine = math.log(prior['9']/n)
        features = Features.get_features_nb(images[j])
        for i in range(len(features)):
            p_zero += math.log(zero[i][features[i]])
            p_one += math.log(one[i][features[i]])
            p_two += math.log(two[i][features[i]])
            p_three += math.log(three[i][features[i]])
            p_four += math.log(four[i][features[i]])
            p_five += math.log(five[i][features[i]])
            p_six += math.log(six[i][features[i]])
            p_seven += math.log(seven[i][features[i]])
            p_eight += math.log(eight[i][features[i]])
            p_nine += math.log(nine[i][features[i]])
        prediction = Bayes.clean_prediction(
            p_zero, p_one, p_two, p_three, p_four, p_five, p_six, p_seven, p_eight, p_nine)
        if prediction == labels[j]:
            # print(prediction, labels[j], 'CORRECT', correct, test_amount)
            correct += 1
        else:
            pass
            # print(prediction, labels[j], 'Incorrect', correct, test_amount)
    return correct/test_amount


def test_bayes_faces(test_amount = 150, train_amount = 451):
    prior, n, face_probs, not_face_probs=train_bayes_faces(train_amount)
    images = Read.read_faces('facedata/facedatatest')
    lables = Read.read_labels('facedata/facedatatestlabels')
    correct = 0
    for j in range(len(lables)):
        p_face = math.log(prior['1']/n)
        p_not_face = math.log(prior['0']/n)
        features = Features.get_face_features_nb(images[j])
        for i in range(len(features)):
            p_face += math.log(face_probs[i][features[i]])
            p_not_face += math.log(not_face_probs[i][features[i]])
        prediction = ''
        if p_face > p_not_face:
            prediction = '1'
        else:
            prediction = '0'
        if prediction == lables[j]:
            correct += 1
            # print(prediction, lables[j], 'CORRECT', correct, test_amount)
        # else:
        #     print(prediction, lables[j], 'Incorrect', correct, test_amount)
    return correct/test_amount


class KNN():
    def classify(test, data, labels, k=5):
        diff = test-data
        dist = (diff ** 2).sum(axis=1) ** 0.5
        sort_dist = dist.argsort()
        knn = {}

        for i in range(k):
            label2 = int(labels[sort_dist[i]])
            knn[label2] = knn.get(label2, 0) + 1

        max_num = 0
        res = 0
        for i in knn:
            if knn[i] > max_num:
                max_num = knn[i]
                res = i
        return res


def train_knn(train_amount, data_set):
    if data_set == 'digits' or data_set == 'digit':
        labels = Read.read_labels('digitdata/traininglabels')
        images = Read.read_digits('digitdata/trainingimages')
        # print(len(train_images))
        train_mat = np.zeros((train_amount, 588))
        for i in range(train_amount):
            train_mat[i] = Features.get_features_knn(images[i])
    else:
        labels = Read.read_labels('facedata/facedatatrainlabels')
        images = Read.read_faces('facedata/facedatatrain')
        # print(len(train_images))
        train_mat = np.zeros((train_amount, 25))
        for i in range(train_amount):
            Features.get_face_features_knn2(images[i])
            train_mat[i] = Features.get_face_features_knn2(images[i])

    return labels[:train_amount], train_mat


def test_knn(train_amount, test_amount, data_set):

    # print(train_labels)
    correct = 0

    # TEST DIGITS
    if data_set == 'digits' or data_set == 'digit':
        test_images = Read.read_digits('digitdata/testimages')
        test_labels = Read.read_labels('digitdata/testlabels')
        train_labels, train_feat = train_knn(train_amount, data_set)

        for i in range(test_amount):
            test_features = Features.get_features_knn(test_images[i])
            prediction = KNN.classify(test_features, train_feat, train_labels)
            # print(' Prediction:', prediction, " Actual:", test_labels[i])
            if prediction == int(test_labels[i]):
                correct += 1

    # TEST FACE DATA SET
    else:
        test_images = Read.read_faces('facedata/facedatatest')
        test_labels = Read.read_labels('facedata/facedatatestlabels')
        train_labels, train_feat = train_knn(train_amount, data_set)

        for i in range(test_amount):
            test_features = Features.get_face_features_knn2(test_images[i])
            prediction = KNN.classify(test_features, train_feat, train_labels)
            if prediction == int(test_labels[i]):
                correct += 1
    return correct/test_amount

    # else:
    #     train_images = Read.read_faces('facedata/facedatatrain')
    #     labels= Read.read_labels('facedata/facedatatrainlabels')
    #     sample_labels = []
    #     for i in range(train_amount):
    #         features = Features.get_face_features_nb(train_images[i])
    #         sample_labels.append(labels[i])

    #     test_images = Read.read_faces('facedata/facedatatest')
    #     test_labels= Read.read_labels('facedata/facedatatestlabels')


# method = input(
#     "what method would we like to do (perceptron[1], bayes[2], knn[3])?: ")
# data_set = input(
#     "what data set would we like to implement this method on (digit[1] or faces[2]): ")

# if method == 'perceptron':
    # if data_set== 'digits' or data_set=='digit':
    #     train_size=int(input("how much data would you like to train on? (max 5000)"))
    #     if train_size > 5000:
    #         train_size=5000
    #     weights=train_perceptron('digits', train_size)
    #     print('max values to test on is 1000')
    #     test_size = int(input("how many values would you like to test on: "))
    #     percentage=test_perceptron(weights,'digits',int(test_size))
    #     print("Correct percentage / Accuracy is: ", percentage)
#     else:
#         train_size=input("how much data would you like to train on? (max 451)")
#         if int(train_size) > 451:
#             train_size=451
#             print("training size too large, using max")
#         print("training.....")
#         weights=train_perceptron("faces")
#         print('max values to test on is 150')
#         test_size = int(input("how many values would you like to test on: "))
#         percentage=test_perceptron(weights,'faces',int(test_size))

# prior, n, face_probs, not_face_probs=train_bayes_faces('faces', 451)
# percentage = test_bayes_faces(prior,n,face_probs,not_face_probs, 151)
# prior_dict, training_number, zero,one,two,three,four,five,six,seven,eight,nine= train_bayes_digits(4000)
# new_percentage=test_bayes_digits(prior_dict,training_number,zero,one,two,three,four,five,six,seven,eight,nine,1000)
# print('\n','\n',new_percentage)

# print(test_knn(1000, 700, 'digits'))
# print(test_knn(2000, 700,'digits'))
# print(test_knn(3000, 700,'digits'))
# print(test_knn(4000, 700,'digits'))
# print(test_knn(451, 100, 'face'))
# print(test_knn_digits(5000, 700,'digit'))
# images = Read.read_digits('digitdata/trainingimages')
# # i = 0
# a = []
# for i in range(10):
#     a.append(images[i])

# for image in a:
#     features = Features.get_features_knn(image)
# mat = Features.string_to_matrix(image)
# count = 0
# line = 0
# for i in range(len(mat)-1):
#     for j in range(len(mat[0])):
#         if mat[i][j] == '+' or mat[i][j] == '#':
#             print('1',end="")
#         else:
#             print('0',end="")
#         count += 1
#     line += 1
#     print()

# print('%d'%count)
# print('%d'%line)


# print(len(Read.read_faces('facedata/facedatatrain')))
# print(len(Read.read_labels('facedata/facedatatrainlabels')))
# print(len(test_labels))

max_train_digits = 5000
max_train_face = 451
test_face = 150
test_digit = 1000

print(f"\n\nPerception Algorithm -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
print(f"   (1) Digits data points:")
start_time = time.time()
for i in range(5):
    percent = int(((i+1) / 5)*100)
    res = test_perceptron('digits', test_digit , int(max_train_digits * (i+1)/ 5))
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")

print(f"      - run time : {time.time()-start_time}")

start_time = time.time()
print(f"\n   (2) Faces data points:")
for i in range(5):
    percent = int(((i+1) / 5)*100)
    res = test_perceptron('face', test_face, int(max_train_face * (i+1)/ 5))
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")
print(f"      - run time : {time.time()-start_time}")

print(f"\nNaive Bayes Classifier -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

start_time = time.time()
print(f"   (1) Digits data points:")
for i in range(5):
    percent = int(((i+1) / 5)*100)
    res = test_bayes_digits(test_digit , int(max_train_digits * (i+1)/ 5))
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")
print(f"      - run time : {time.time()-start_time}")
print(f"\n   (2) Faces data points:")
start_time = time.time()
for i in range(5):
    percent = int(((i+1) / 5)*100)
    res = test_bayes_faces(test_face , int(max_train_face * (i+1)/ 5))
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")
print(f"      - run time  : {time.time()-start_time}")

print(f"\nK- Nearest Neighbors -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
start_time = time.time()
print(f"   (1) Digits data points:")
for i in range(5):
    percent = int(((i+1) / 5)*100)
    # print(percent)
    res = test_knn(int(max_train_digits * (i+1) / 5), test_digit, 'digits')
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")
print(f"      - run time : {time.time()-start_time}")

print(f"\n   (2) Faces data points:")
start_time = time.time()
for i in range(5):
    percent = int(((i+1) / 5)*100)
    res = test_knn(int(max_train_face * (i+1)/ 5), test_face, 'face')
    print(f"      - Train on {percent}% data points: {res * 100:.2f}% of accuracy")
print(f"      - run time : {time.time()-start_time}")


labels = Read.read_labels('facedata/facedatatrainlabels')
images = Read.read_faces('facedata/facedatatrain')


Features.get_face_features_knn2(images[0])
