import os
import numpy as np
import FisherFace
from numpy import linalg
from matplotlib import pyplot as plt



# compute the accuracy
def ComputeAccuracy(num, num_sample, confusion_mat):
    
    trace = 0
    for i in range(num):
        trace += confusion_mat[i][i]

    accuracy = trace / num_sample
    return accuracy


def classifier(template, feature):
    
    [row, col] = template.shape
    dst_min = float('Inf')
    
    for i in range(col):
        dst = linalg.norm((template[:, i : i + 1] - feature).T)
        if dst < dst_min:
            dst_min = dst
            label = i
    return label


def process_pca(faces, num, K, idLabel):
    W, LL, m = FisherFace.myPCA(faces)
    We = W[:, 5:K + 5] # select features
    me = m[:, np.newaxis]
    Mat = np.arange(0).reshape(num, 0)
    Mat = Mat.tolist()

    for i in range(len(idLabel)):
        x = faces[:, i : i + 1]
        y = np.dot(We.T, (x - me))
        y = y.T
        Mat[idLabel[i]].extend(y)

    Z = np.ones((num, y.size))

    for i in range(num):
        mat = np.array(Mat[i])
        Z[i] = np.mean(mat.T, 1)


    Z = Z.T
    return Z, W, me, We

def get_PCA_feature(x,We,me):
    
    feature = np.dot(We.T, (x - me))
    
    return feature


def process_lda(faces, W, me, idLabel):

    Xmat = []
    for i in range(len(idLabel)):
        x = faces[:,i:i+1]
        x = np.dot(W.T, (x - me))
        Xmat.extend(x.T)
    Xmat = np.array(Xmat)
    X_Trans = Xmat.transpose()
    
    LDAW,Centers, classLabels = FisherFace.myLDA(X_Trans,idLabel)

    return Centers,LDAW,me

def get_LDA_feature(x, LDAW, We, me):
    
    feature = np.dot(LDAW.T, np.dot(We.T, (x - me)))
    
    return feature


def process_fusion(feature1_template, feature2_template, num, alpha):
    
    [row1, col1] = feature1_template.shape
    [row2, col2] = feature2_template.shape
    
    fusion = np.zeros(((row1 + row2), num))
    
    for i in range(num):
            y1 = feature1_template[:, i : i + 1]
            y2 = feature2_template[:, i : i + 1]
            fusion[:, i : i + 1] = np.vstack((np.dot(alpha, y1), np.dot((1 - alpha), y2)))

    fusion = np.array(fusion)            

    return fusion

def get_fusion_feature(feature1, feature2, alpha):
    
    feature = np.vstack((np.dot(alpha, feature1), np.dot((1- alpha), feature2)))
    
    return feature

def main():
    K = 30
    K1 = 90
    num = 10
    faces, idLabel = FisherFace.read_faces('./train')

    Z ,W, me, We = process_pca(faces, num, K, idLabel)
    faces_test, idLabel_test = FisherFace.read_faces('./test') 
    N = len(idLabel_test)
    confusion_mat = np.zeros((num, num))
    confusion_mat = np.uint8(confusion_mat)

    print('PCA Confusion Matrix:')
    for i in range( N ) :
        feature = get_PCA_feature(faces_test[:, i : i + 1],We,me)
        label = classifier(Z, feature)
        confusion_mat[idLabel[i]][label] += 1
    print(confusion_mat)
    accuracy = ComputeAccuracy(num, N, confusion_mat)
    print('PCA Accuracy: ',accuracy)

    W1 = W[:, : K1]
    centers, W_lda, me = process_lda(faces, W1, me, idLabel)
    confusion_mat[:][:] = 0 
    print('LDA Confusion Matrix:')
    for i in range( N ) :
        feature = get_LDA_feature(faces_test[:,i:i+1], W_lda, W1, me)
        label = classifier(centers, feature)
        confusion_mat[idLabel[i]][label] += 1
    print(confusion_mat)
    accuracy = ComputeAccuracy(num, N, confusion_mat)
    print('LDA Accuracy: ', accuracy)

    fusion = np.zeros((num, 0))
    fusion = fusion.tolist()
    plt.figure()
    plt.xlabel('alpha')
    plt.ylabel('Accuracy Rate')
    xaxes = []
    yaxes = []
    for alpha in range(1,10):
        alpha = alpha/10
        fusion = process_fusion(Z, centers, num, alpha)
        confusion_mat[:][:] = 0
        print('Fusion Confusion Matrix:')
        for i in range( N ) :
            feature_PCA = get_PCA_feature(faces_test[:, i : i + 1], We, me)
            feature_LDA = get_LDA_feature(faces_test[:, i : i + 1], W_lda, W1, me)
            feature = get_fusion_feature(feature_PCA, feature_LDA, alpha)
            label = classifier(fusion,feature)
            confusion_mat[idLabel[i]][label] += 1
        print(confusion_mat)
    
        accuracy = ComputeAccuracy(num, N, confusion_mat)
        xaxes.append(alpha)
        yaxes.append(accuracy)
        print("Fusion Accuracy: ",accuracy," alpha: ",alpha)
        plt.annotate('(' + str(round(alpha, 2)) + ', ' + str(round(accuracy, 2)) + ')', (alpha, accuracy))
    plt.plot(xaxes, yaxes, 'b-o')
    print(xaxes, yaxes)
        

        
    plt.savefig("figure.png")
    plt.show()

if __name__ == '__main__':
    main()