import math


test_cases_p_word_given_class = {'opinions': 0.0018821,
'yet': 0.003764,
'away': 0.0018821,
'interested': 0.0025094,
'extended': 0.0018821,
'married': 0.0018821,
'received': 0.0018821,
'adapted': 0.0018821,
'remainder': 0.0018821,
'behind': 0.0018821,
'pain': 0.0025094,
'old': 0.0025094,
'private': 0.0018821,
'solicitude': 0.0031368,
'examine': 0.0012547,
'removal': 0.0025094,
'unaffected': 0.0012547,
'late': 0.0018821,
'sold': 0.003764,
'UNKNOWN_WORD': 0.0006274}


def test_compute_p_word_given_class():
    from probabilities import compute_p_word_given_class

    train_paths = ['data/test_text/test_text_' + str(i) + '.txt' for i in range(1,6)]
    student_solution = compute_p_word_given_class(train_paths, 100)

    test_cases = test_cases_p_word_given_class

    count_correct = 0
    for word in test_cases:
        if math.isclose(student_solution[word], test_cases[word], abs_tol = 1e-6):
            count_correct+=1

    result = str(count_correct) + ' out of ' + str(len(test_cases)) + ' correct'
    print(result)


def test_compute_p_class():
    from probabilities import compute_p_class

    test_cases = {(1,1): 0.5, (1,2): 0.4, (2,1): 0.6, (2,2): 0.5, (3,3): 0.5, (4,2): 0.625, (4,4): 0.5}

    count_correct = 0
    for i,j in test_cases:
        if compute_p_class(i, j) == test_cases[(i,j)]: count_correct += 1

    result = str(count_correct) + ' out of ' + str(len(test_cases)) + ' correct'
    print(result)


def test_compute_p_class_given_input():
    from probabilities import compute_p_class_given_input

    test_paths = ['data/test_text/test_text_' + str(i) + '.txt' for i in range(1,6)]
    test_cases = [-2012.09438, -2216.4758, -2088.3987, -2266.9948, -2360.9622]

    p_wc = test_cases_p_word_given_class
    p_c = 1

    count_correct = 0
    for i, test_path in enumerate(test_paths):
        if math.isclose(compute_p_class_given_input(test_path, p_wc, p_c), test_cases[i], abs_tol = 1e-3): 
            count_correct += 1

    result = str(count_correct) + ' out of ' + str(len(test_cases)) + ' correct'
    print(result)



if __name__ == '__main__':
    print('compute_p_word_given_class')
    test_compute_p_word_given_class()

    print('\ncompute_p_class')
    test_compute_p_class()

    print('\ncompute_p_class_given_input')
    test_compute_p_class_given_input()