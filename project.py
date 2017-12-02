import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join
import cv2
import imageio
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def main():
   imagesDir = 'katkam-scaled'
   weatherDataDir = 'yvr-weather'
   # Used this stackoverflow hint to get all the names of files in directory
   #https://stackoverflow.com/questions/33369832/read-multiple-images-on-a-folder-in-opencv-python
   imageFileNames = [ f for f in listdir(imagesDir) if isfile(join(imagesDir,f)) ]
   weatherDataFileNames = [ f for f in listdir(weatherDataDir) if isfile(join(weatherDataDir,f)) ]
   
   # Read in the first weather data csv file
   weatherData = pd.read_csv(join(weatherDataDir, weatherDataFileNames[0]), usecols=[1, 2, 3, 4, 6, 24], skiprows=np.linspace(0, 15))

   # Join every other weather data csv file onto the dataframe made from the first one, then get rid of null values
   for n in range(1, len(weatherDataFileNames)):
      weatherData = weatherData.append(pd.read_csv(join(weatherDataDir, weatherDataFileNames[n]), usecols=[1, 2, 3, 4, 6, 24], skiprows=np.linspace(0, 15)))
   
   # This only records weather entries that have a description for the weather type
   weatherDataCleaned = weatherData.dropna().reset_index(drop=True)
    
   # Assign each day an ID, which we are going to use to check against the list of image files if there is an image that matches that particular day and time
   weatherDataCleaned['ID'] = weatherDataCleaned['Year']*1e6 + weatherDataCleaned['Month']*1e4 + weatherDataCleaned['Day']*1e2 + weatherDataCleaned['Time'].str.slice(0, 2).astype(float)
   weatherDataCleaned['ID'] = weatherDataCleaned['ID'].astype(int)
    
   # Preallocate the space for all of our images
   actualImages = np.empty([weatherDataCleaned.shape[0], 192, 256, 3], dtype=np.uint8)
   
   # For every weather recording, check if there is a corresponding image for that time and day
   numImages = 0
   imageIdx = np.full(weatherDataCleaned.shape[0], -1)
   for n in range(0, weatherDataCleaned.shape[0]):
      filename = "katkam-" + str(weatherDataCleaned['ID'][n]) + "0000" + ".jpg"
      if (filename in imageFileNames):
         actualImages[numImages, :, :, :] = imageio.imread(join(imagesDir, filename))
         imageIdx[n] = numImages
         numImages = numImages+1
   
   # The imageIdx keeps treck of where in our array of images the image for a particular data entry is
   weatherDataCleaned['imageIdx'] = imageIdx
   
   # Delete data entries with no corresponding image
   weatherDataCleaned = weatherDataCleaned[weatherDataCleaned['imageIdx'] >= 0]
   # Trim the size of our image array down to however many images we actually read in
   actualImages = actualImages[ 0:numImages, :, :, :]
   
   # We're only interested in the portion of the image forming the sky, so trim out the cityscape
   # Sidenote: Analysis could be done on the city portion as well, especially to determine rain, but I decided not to go that route
   skyImages = actualImages[:, 0:100, :, :]
   
   # The first measure used to analyze the weather is the colour
   # If the sky is bluer, we can expect it to be clearer
   # The other two colors don't add much accuracy, but they help a bit
   # We find the colorfulness by dividing the sum of the blueness of every pixel by the total brightness, the value will be between 0 and 1
   # So a pixel of rgb value 100/100/200 will have a blueness of 200/400 = .5
   # A colorless pixel will have R=G=B, for example 150/150/150, for a blueness of .33
   weatherDataCleaned['Blueness'] = (skyImages[:, :, :, 2].sum(axis=(1, 2)))/(skyImages[:, :, :, :].sum(axis=(1, 2, 3)))
   weatherDataCleaned['Greeness'] = (skyImages[:, :, :, 0].sum(axis=(1, 2)))/(skyImages[:, :, :, :].sum(axis=(1, 2, 3)))
   weatherDataCleaned['Redness']  = (skyImages[:, :, :, 1].sum(axis=(1, 2)))/(skyImages[:, :, :, :].sum(axis=(1, 2, 3)))

   # We also use the overall brightness of an image to help guess the weather
   weatherDataCleaned['Brightness'] = (skyImages[:, :, :, :].sum(axis=(1, 2, 3)))
   
   # The two weathers I was having the most trouble telling apart were cloudy, and mostly clear
   # Both these weather types have some clouds in the sky, this is a method to give some metric of how many clouds there are
   # Taking the gradient of an image is one method of finding lines in the image. An image with more lines means more small patchy clouds,
   # whereas less lines means either smooth clouds, or clear sky.
   # Summing the entire gradient of the image gives us a measure of the "patchiness" of the image, relative to others
   
   # The gradient makes sense when we take it from greyscale images, so convert our images to greyscale before we do the calculations
   greyImages = np.uint8((actualImages[:, :, :, 0].astype(np.float) + actualImages[:, :, :, 1].astype(np.float) + actualImages[:, :, :, 2].astype(np.float))/3)
   gradients = np.zeros(np.shape(greyImages), dtype=np.int16)
   for n in range(0, np.shape(greyImages)[0]):
      # cv2.Laplacian calculates the gradients of images using the laplacian kernel, we take the absolute value because we don't care if the change in brightness
      # is up or down, just its magnitude
      gradients[n, :, :] = np.absolute(cv2.Laplacian(greyImages[n, :, :], ddepth=cv2.CV_16S))
       
       
   weatherDataCleaned['Gradient'] = (gradients[:, :, :].sum(axis=(1, 2)))
   
   # Now we do some additional data wrangling to best do our training and testing
   
   # Many pictures were taken at night when it was very dark, those pictures are just all black and pollute the data, so get rid of them
   weatherDataReady = weatherDataCleaned[weatherDataCleaned['Brightness'] > 10000000]
   
   # Many groups were very similar, or just had a few entries, so I decided to just try and tell the difference between 4 categories: Precipitating, Clear, Cloudy, and Mostly Clear
   # Stuff like snow had such a tiny amount of training data, it was just being ingnored entirely for the tests (every snow scene was being classified as something else)
   groupPrecipitating = ('Drizzle', 'Drizzle, Fog', 'Heavy Rain Showers,Moderate Snow Pellets,Fog', 'Heavy Rain,Fog', 
                'Heavy Rain,Moderate Hail,Fog', 'Moderate Rain', 'Moderate Rain Showers', 'Moderate Rain Showers,Fog',
                'Moderate Rain,Drizzle', 'Moderate Rain,Fog', 'Rain', 'Rain Showers', 'Rain Showers,Fog', 
                'Rain Showers,Snow Pellets', 'Rain Showers,Snow Showers', 'Rain Showers,Snow Showers,Fog', 
                'Rain,Drizzle', 'Rain,Drizzle,Fog', 'Rain,Fog', 'Rain,Snow,Fog', 'Thunderstorms', 'Drizzle,Fog', 
                'Snow', 'Snow Showers', 'Snow,Fog', 'Snowing', 'Rain,Snow', 'Moderate Snow')
   groupCloudy = ('Mostly Cloudy', 'Cloudy', 'Fog', 'Freezing Fog')
   
   weatherDataGrouped = weatherDataReady
   
   # Group the weather data based on previously defined groups
   pd.options.mode.chained_assignment = None
   weatherDataGrouped.loc[weatherDataGrouped['Weather'].isin(groupPrecipitating), 'Weather'] = 'Precipitating'
   weatherDataGrouped.loc[weatherDataGrouped['Weather'].isin(groupCloudy), 'Weather'] = 'Cloudy'
   
   # These are the parameters of the images we are doing our guessing with
   parameters = weatherDataGrouped[['Blueness', 'Redness', 'Greeness', 'Brightness', 'Gradient']].values
   
   # Change shape of names so the errors stop
   names = np.ravel(weatherDataGrouped[['Weather']].values)
   
   # Split into training and testing data
   images_train, images_test, names_train, names_test = train_test_split(parameters, names)
   
   print("Total number of images used is " + str(weatherDataGrouped.count()[0]))
   
   # Scaling is important with the color values especially, since although they range from 0-1, they tend to cluster between 25-40%
   # SCV performed the best of common ML techniques at the start of the project, but after tuning and data cleaning,
   # the neural network tended to do better by a couple percent points
   myModel1 = make_pipeline(
      StandardScaler(),
      SVC(C=2))
   myModel1.fit(images_train, names_train)
   
   print('Score for SVM is: ')
   print(myModel1.score(images_test, names_test))
   
   countWeather = weatherDataGrouped.groupby('Weather')
   weatherList = countWeather.agg('count')
   priors = [weatherList.iloc[0][0], weatherList.iloc[1][0], weatherList.iloc[2][0], weatherList.iloc[3][0]]/weatherList.sum()[0]
   myModel2 = make_pipeline(
      StandardScaler(),
      GaussianNB(priors=priors))
   myModel2.fit(images_train, names_train)
   
   print('Score for GaussianNB is: ')
   print(myModel2.score(images_test, names_test))
   
   myModel3 = make_pipeline(
      StandardScaler(),
      KNeighborsClassifier(n_neighbors=10))
   myModel3.fit(images_train, names_train)
   
   print('Score for KNN is: ')
   print(myModel3.score(images_test, names_test))
   
   myModel4 = make_pipeline(
      StandardScaler(),
      MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 3),
                      activation='logistic'))
   myModel4.fit(images_train, names_train)
   
   print('Score for Neural Networks is: ')
   print(myModel4.score(images_test, names_test))
   
   
if __name__ == '__main__':
    main()