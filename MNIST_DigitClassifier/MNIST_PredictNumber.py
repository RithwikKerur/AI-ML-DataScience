import joblib
import pygame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

digit_clf = joblib.load(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\KNN_MNIST_ImageClassifier_v2")
def guessNum (data):
    return digit_clf.predict(data), digit_clf.predict_proba(data)


def main():
    running = True
    while (running):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if (keys[pygame.K_RETURN] == True):
                    pygame.image.save(screen, r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image = Image.open(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image = image.resize((28, 28), Image.ANTIALIAS)
                    image.save(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image_arr = np.array(image)
                    data = [[]]
                    for x in image_arr:
                        for z in x:
                            intensity = (int(np.mean(z))) + 70
                            if (intensity < 85):
                                intensity = 0
                            else:
                                if (intensity > 255):   
                                    intensity = 255
                            data[0].append(intensity)
                    prediction, pred_accuracy = guessNum(data)
                    pred_accuracy = np.array(pred_accuracy).round(3)
                    print (pred_accuracy)
                    print ("Predicted number: " + str(prediction))
                    # image_28x28 = np.array(data).reshape(28,28)
                    # plt.imshow(image_28x28, cmap = "gray")
                    # plt.show()
            if (pygame.mouse.get_pressed()[0] == True ):
                mouse_pos = pygame.mouse.get_pos()
                pygame.draw.circle(screen, WHITE, mouse_pos, width)
            if (pygame.mouse.get_pressed()[2] == True):
                screen.fill(BLACK)
        pygame.display.update()


pygame.init()
WHITE = (255,255,255)
BLACK = (0,0,0)
height = 350
screen = pygame.display.set_mode((height,height))
pygame.display.set_caption("Draw Number")
width = height // 40
main()
pygame.quit()

