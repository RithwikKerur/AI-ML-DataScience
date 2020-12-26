import joblib
import pygame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image

# Load the saved model generated with the Jupyter notebook (MNIST_DigitClassifier_v2.ipynb)
digit_clf = joblib.load(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\KNN_MNIST_ImageClassifier_v2")
def guessNum (data):

    # Make predictions with the model, return the prediction and the associated prediction probability
    return digit_clf.predict(data), digit_clf.predict_proba(data)

# Main loop for the graphics for the number drawing
def main():
    running = True
    while (running):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # If the user presses ENTER/RETURN, then take a screenshot of the current image, resize it to a 28x28 image,
            # and then convert it to an 1 by 784 (28 * 28) flattened array containing the intensities of each pixel in the
            # image drawn by the user.
            if (keys[pygame.K_RETURN] == True):
                    pygame.image.save(screen, r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image = Image.open(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image = image.resize((28, 28), Image.ANTIALIAS)

                    ## Save the image to see what the resized image looks like before any preprocessing
                    # image.save(r"C:\Users\408aa\Desktop\Python\DataScience\MNIST_DigitClassifier\number.jpeg")
                    image_arr = np.array(image)
                    data = [[]]
                    for x in image_arr:
                        for z in x:

                            # Pygame's drawing combined with resizing results in lighter final image, 
                            # artificially increase the intensity of the image (by 70), so it will produce a better prediction result.
                            intensity = (int(np.mean(z))) + 70
                            
                            # Remove the noise in the resized image, if the pixel intensity is still less than 85 (15 originally), 
                            # then just make it a pixel with 0 intensity.
                            if (intensity < 85):
                                intensity = 0
                            else:

                            # If the intensity is greater than 255 (the most intense value of a pixel), then just set it to 255
                            # this will make it more similar to the training data and closer to the data points in the training data
                            # (remember the model is trained with a KNN classifier)
                                if (intensity > 255):   
                                    intensity = 255
                            data[0].append(intensity)
                    prediction, pred_accuracy = guessNum(data)
                    pred_accuracy = np.array(pred_accuracy).round(3)
                    print (pred_accuracy)
                    print ("Predicted number: " + str(prediction))

                    ## Produce a gray scaled image of the translation (after denoising and intensifying) on a Pyplot for easy
                    ## visualization of how noisy the resized image is after preprocessing
                    # image_28x28 = np.array(data).reshape(28,28)
                    # plt.imshow(image_28x28, cmap = "gray")
                    # plt.show()
                    
            # Draw on the screen if the mouse 1 button (left click) is pressed
            if (pygame.mouse.get_pressed()[0] == True ):
                mouse_pos = pygame.mouse.get_pos()
                pygame.draw.circle(screen, WHITE, mouse_pos, penWidth)
            
            # Erase (make the screen black) if the mouse 2 button (right click) is pressed.
            if (pygame.mouse.get_pressed()[2] == True):
                screen.fill(BLACK)
        pygame.display.update()


pygame.init()
WHITE = (255,255,255)
BLACK = (0,0,0)

# Height of the display screen
height = 350
screen = pygame.display.set_mode((height,height))
pygame.display.set_caption("Draw Number")

# Width of the stylist/pencil that the user uses to draw on the screen (tried to make the ratio of screen size
# to pencil/stylist width as close to the original MNIST data as possible to produce the most accurate predictions
# generally, the thinner the pencil width, the worse the predictions.
penWidth = height // 17
main()
pygame.quit()

