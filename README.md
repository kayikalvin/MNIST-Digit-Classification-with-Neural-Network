# MNIST-Digit-Classification-with-Neural-Network



## Deep Learning Image Prediction Workflow
This code snippet demonstrates the process of predicting a handwritten digit from an image using a deep learning model. The following steps outline the operations performed, along with the reasons for each step and how they contribute to the overall prediction process.

1. Input Image Path

``` Python
input_image_path = input('Path of the image to be predicted: ')
```
 **I** Purpose: Prompt the user to input the file path of the image to be predicted.<br>
 **II** Explanation: The user provides the path to the image (e.g., image.png). This path is essential for loading 
        the image that the model will use for prediction.
2. Read the Image
``` Python
input_image = cv2.imread(input_image_path)
```
 **I** Purpose: Read the image file from the provided path.<br>
 **II** Explanation: The cv2.imread() function from OpenCV reads the image file and loads it into memory as a NumPy array. The image is typically in RGB (or BGR) format, depending on how OpenCV handles it.
3. Display the Image
``` Python
cv2_imshow(input_image)
```
 **I** Purpose: Display the input image for visual inspection.<br>
 **II** Explanation: The cv2_imshow() function (often used in Jupyter notebooks or Google Colab) is used to show           the image to the user. This allows for confirmation of the image before prediction. In a standard Python           script, cv2.imshow() would be used.
4. Convert Image to Grayscale
``` Python
grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
```
 **I** Purpose: Convert the image to grayscale to simplify the input.<br>
 **II** Explanation: Handwritten digit classification models usually perform better on grayscale images since color is not essential for digit recognition. The cv2.cvtColor() function converts the image from RGB (or BGR) to grayscale.
5. Resize the Image
``` Python
input_image_resize = cv2.resize(grayscale, (28, 28))
```
 **I** Purpose: Resize the image to 28x28 pixels to match the model’s input size.<br>
 **II** Explanation: Many digit recognition models (e.g., MNIST) expect the input images to be of size 28x28 pixels. This step ensures that the image is resized to the correct dimensions.
6. Normalize the Image
``` Python
input_image_resize = input_image_resize / 255
```
 **I** Purpose: Normalize the pixel values to the range [0, 1].<br>
 **II** Explanation: Deep learning models perform better when input data is scaled. The pixel values range from 0 to 255, so dividing by 255 converts the values to the range [0, 1], which helps the model process the image more efficiently.
7. Reshape the Image
``` Python
image_reshaped = np.reshape(input_image_resize, [1, 28, 28])
```
 **I** Purpose: Reshape the image to match the input shape the model expects.<br>
 **II** Explanation: Deep learning models typically expect input in batches. Even though there is only one image, it needs to be reshaped into a batch of size 1. The model likely expects a shape of [batch_size, height, width], so the image is reshaped to [1, 28, 28].
8. Make the Prediction
``` Python
input_prediction = model.predict(image_reshaped)
```
 **I** Purpose: Make a prediction on the reshaped image using the trained model.<br>
 **II** Explanation: The reshaped image is passed through the model to obtain the prediction. The model typically outputs a vector of probabilities, one for each possible class (in this case, digits 0-9).
9. Get the Predicted Label
``` Python
input_pred_label = np.argmax(input_prediction)
```
 **I** Purpose: Extract the class with the highest probability from the model's output.<br>
 **II** Explanation: The model’s output is a vector of probabilities, one for each possible digit (0-9). The np.argmax() function finds the index of the maximum value in the prediction vector, which corresponds to the predicted label (digit).
10. Print the Prediction
``` Python
print('The Handwritten Digit is recognised as ', input_pred_label)
```
 **I** Purpose: Output the predicted digit label.
 **II** Explanation: This line prints the predicted digit (from 0 to 9), providing the user with the result of the model’s prediction.

