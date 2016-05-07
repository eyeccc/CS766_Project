from openpyxl import load_workbook
import random

def get_label_from_01(b):
    if b == 1:
        return 'Vangogh'
    elif b == 2:
        return 'Gauguin'
    elif b == 3:
        return 'Braque'
    elif b == 4:
        return 'Gris'
    elif b == 5:
        return 'Monet'
    elif b == 6:
        return 'Raphael'
    elif b == 7:
        return 'Titian'
    else:
        print 'Error get_01_from_label'
        return None


def generate_targetFeature(file_name):
    features = []
    labels = []
    imageName = []
    lines = readExcel(file_name)
    random.seed(1234567)
    random.shuffle(lines)
    for line in lines:
        feature = []
        if line[37] == 1 or line[37] == 4:
            for i in range(0,37):
                feature.append(line[i])
            features.append(feature)
            labels.append(line[37])
            imageName.append(line[38])
    print labels

    return features, labels, imageName


def generate_feature(file_name):
    features = []
    labels = []
    imageName = []
    lines = readExcel(file_name)
    random.seed(1234567)
    random.shuffle(lines)
    for line in lines:
        feature = []
        for i in range(0,37):
            feature.append(line[i])
        features.append(feature)
        labels.append(line[37])
        imageName.append(line[38])

    #print labels


    return features, labels, imageName


def readExcel(file_name):
    wb = load_workbook(filename=file_name, read_only=True)
    ws = wb.get_sheet_by_name(name = 'Sheet1')
    rows = ws.rows
    columns = ws.columns
    content = []
    for row in rows:
        line = [col.value for col in row]
        #print line
        content.append(line)

    return content
