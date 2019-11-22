import os

def get_words(message):
    words_tags = message.split('\n')
    words_raw = [pair.split()[0].lower() for pair in words_tags if pair.split() != [] and len(pair.split()[0]) > 1]    
    return words_raw


def open_file(file_name):
    content = ''
    found_start = False
    with open(file_name, 'r', encoding='Latin-1') as handle:
        for line in handle:
            if line == '\n':
                found_start = True
            if found_start:
                content += line
    return content


def get_data_paths(path, test_data_start, numfolds, fold):
    train_files = []
    test_files = []
    files = os.listdir(path)
    for filename in files:
        filenum = int(filename.split('.')[0][2:].split('_')[0])
        if filenum % numfolds == fold:
            test_files.append(path + '/' + filename)
        else:
            train_files.append(path + '/' + filename)
    return train_files, test_files