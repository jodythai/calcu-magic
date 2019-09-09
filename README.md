# "Magic" Calculator

![](https://i.imgur.com/Vq7jj8b.png)

## Introduction
This weekly project is to calculate a given expression. There are two datasets needed to achieve the goal. The first one is MNIST dataset which contains 70,000 images. The second one is the Operators dataset consists of 62,400 images for +, -, *, and / operators. The images in these two datasets are specially selected for classifying. Finally, we will build a Flask app to help user capture the mathematic expression or write the expression on canvas then provide the result.

## Our team members
* Thai Binh Duong [github](https://github.com/jodythai)
* Nguyen Thi Diem Hang [github](https://github.com/hangnguyennn13)
* Ho Huy Hoa [github](https://github.com/hoahh2201)


## Datasets
### MNIST dataset
For handwritten recognition task, we use MNIST dataset.
![](https://i.imgur.com/hwAtC38.png)

### Operators dataset
We used a dataset from [Kaggle.com](https://www.kaggle.com/xainano/handwrittenmathsymbols)
## Project goal
* Build CNN model to classify digits and operators. Moreover the model has to reach more than 90% accuracy score
* Build a Flask app to calculate the mathematic expression provided by the user with the following requirements:
    * Support operators: +, -, *, /
    * Equation format: `<number> <operator> <number>`
    * Support multiple numbers and operators

## Understanding the dataset
#### MNIST dataset: Flattened image which have shape of 28*28 pixels
* Label: 0 - 9
* Pixel0 - Pixel783: the value of a specific pixel, range from 0 - 255
#### Operators dataset
* Label: 
    * '+' : 0
    * '-' : 1
    * '/' : 2
    * 'x' : 3
* Image path: path to a specific image

### Link to the flask app
* [https://github.com/jodythai/calcu-magic](https://github.com/jodythai/calcu-magic)

### Tasks needed to be done
* Create VM on GCE with Deep learning platform to train model: https://hackmd.io/SEVZeQMJRa2JJMxd9y8PnQ
* Build Model:
    1. Load dataset + check data
        * MNIST dataset
            * Using tensorflow.keras.datasets.mnist
            * Label: 0 - 9
            * Image is in form of numpy array (784,)
        * Operator dataset:
            * The dataset contains 62,400 images.
            * Load only the image path using os and glob libraries
            * Label the image according to its path
                * '+' : 0
                * '-' : 1
                * '/' : 2
                * 'x' : 3 
        
    2. Preprocess data:
        * Convert the train images to grayscale
        * Reshape the train images into 4 dimensions
        * Rescale the train images to values between 0 - 1 
        

    4. Check whether we have a balanced dataset or not using seaborn countplot on the label columns. If not then consider to choose between applying undersamplying or oversampling technique. However, we have a balanced dataset so we can skip this part.
    
    4. Build Model:
        * Model to classify whether the image is operator or digit
        * Model to classify digit
        * Model to classify operator
    5. CNN architecture:
        * **Input Layer:** It represent input image data. It will reshape image into single diminsion array. Example your image is 64x64 = 4096, it will convert to (4096,1) array.

        * **Conv Layer:** This layer will extract features from image.

        * **Pooling Layer:** This layerreduce the spatial volume of input image after convolution.

        * **Fully Connected Layer:** It connect the network from a layer to another layer

        * **Output Layer:** It is the predicted values layer.
        
        * **Loss:**- To make our model better we either minimize loss or maximize accuracy. NN always minimize loss. To measure it we can use different formulas like 'categorical_crossentropy' or 'binary_crossentropy'. Here I have used binary_crossentropy

        * **Optimizer :**- If you know a lil bit about mathematics of machine learning you might be familier with local minima or global minima or cost function. To minimize cost function we use different methods For ex :- like gradient descent, stochastic gradient descent. So these are call optimizers. We are using a default one here which is adam

        * **Metrics :**- This is to denote the measure of your model. Can be accuracy or some other metric.
        
    6. Define a Sequential model for the combined dataset:
        * 3 Conv2d layers, each with 32, 64 and 64 filters and relu activation functions and a (3,3) kernel size, and padding='same'
        * We follow each Conv2d layer with a MaxPooling2d layer with a (2,2) pool size
        * Finally, we have a Dense layer with 512 nodes and relu activation.
        * The output Dense layer has 10 nodes (for the 10 digits) and a softmax activation function (for a a multi-class classification problem)
        * We use the categorical-crossentropy loss and the adam optimizer.
    ```python=
        model = Sequential()
        # add Convolutional layers
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                         input_shape=(image_height, image_width, num_channels)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))    
        model.add(Flatten())
        # Densely connected layers
        model.add(Dense(128, activation='relu'))
        # output layer
        model.add(Dense(num_classes, activation='softmax'))
        # compile with adam optimizer & categorical_crossentropy loss function
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ```
    NOTE: With this CNN, we get a training accuracy of 99.7% (and validation accuracy of 99.3%) after 15 epoches for the combined dataset between mnist and operator
    
    7. Using LinearSVC (SVM) for the third model to classify operators.

* Run the Flask App:
    
    1. Create and activate virtual environment:
    `virtualen -p python3 env`
    `source env/bin/activate`

    2. Install libraries inside the virtual environment:
    `pip install -r requirements.txt`
    
    6. Activate debug mode:
        ```python
        export FLASK_DEBUG=1
        export FLASK_APP=main.py
        export FLASK_ENV=development
        ```
    
    7. Run the flask app:
    `flask run`

