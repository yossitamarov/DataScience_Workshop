import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm
import pathlib
import csv
import os
import zipfile
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.model_selection import train_test_split

# for map graphical view:
import matplotlib.cm
import matplotlib as mpl
from geonamescache import GeonamesCache
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

# for PCA:
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for outliers detections
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
import statsmodels.api as sm

# for images comparison:
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# model for feature selection:
from sklearn import datasets, linear_model, decomposition
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sp
import sklearn.feature_selection as fs
from sklearn import kernel_ridge
# import skfeature as skf

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.grid_search import GridSearchCV

# Imports for kernel ridge:
from sklearn.model_selection import GridSearchCV

# Imports for Results check
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

# cPickle for model fit download to file
import _pickle as cPickle

# loading bar
from ipywidgets import FloatProgress
from IPython.display import display

# Data paths
path_complete_data = os.path.join('merged_data_ready', 'merged_data.csv')
zip_file_path = os.path.join('raw_data', 'DB_Data', 'Edstast_data.zip')
path = os.path.join('raw_data', 'DB_Data', 'Edstast_data.csv')
path_fixed = os.path.join('raw_data', 'DB_Data', 'Edstast_data_fixed.csv')
input_labels = os.path.join('raw_data', 'Labels', 'Happy_Planet_Index_Data')

# Paths for the graphical map visualization use
countries_codes = os.path.join('raw_data', 'DB_Data', 'WDI_Country.csv')
shapefile = os.path.join('map_files', 'ne_10m_admin_0_countries')
template_image = os.path.join('map_files', 'imgfile.png')
globe_plots = 'globe_plots'
uncorrolated_plots = 'uncorrolated_images'

# Dumped models path
dumped_models = 'dumped_models'

# Years with labels
rellevant_years_for_labels = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', \
                              '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', \
                              '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', \
                              '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', \
                              '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2009', '2012', '2016']
rellevant_years = [year + '.00' for year in rellevant_years_for_labels]
classifiers = ['Random_Forest', 'Linear_Regression', 'Lasso', 'Ridge', 'Kernel_Ridge']


class DataPreparation():
    @staticmethod
    def retriveMergedFilePath():
        return path_complete_data

    @staticmethod
    # Merge the data with the labels
    def mergeDataWithLabels(working_frame, labels):
        result = pd.merge(working_frame, labels, how='inner', on=['country', 'year'])
        result.to_csv(path_complete_data)

    @staticmethod
    # Cleaning the CSV Files Out From Commas
    def cleanCommasFromCSV():
        with open(path, "r", newline="") as infile, open(path_fixed, "w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(item.replace(",", "") for item in row)

    @staticmethod
    # Obtain The Labeled Data
    def getDataFrameForLabelCSV(path, year):
        df = pd.read_csv(path, skiprows=0, usecols=[1, 8])
        df.loc[:, 'year'] = pd.Series(float(rellevant_years[rellevant_years_for_labels.index(year)]), index=df.index)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        df.drop(df.index[[0]])
        return df

    @staticmethod
    # Take Data From DataSets
    def extractDataFromCSV():
        df = pd.read_csv(path, header=None, skiprows=0, encoding='iso-8859-1')
        df.drop(df.columns[[1, 3]], axis=1, inplace=True)
        df.loc[0, 0] = 'country'
        df.columns = df.loc[0]
        df = pd.pivot_table(df, index='country', columns='Indicator Name')
        df = df.stack(level=0)
        df.reset_index(inplace=True)
        df.rename(columns={0: 'year'}, inplace=True)
        df.rename(columns={"Indicator Name": 'series'}, inplace=True)
        df.to_csv(path_complete_data, encoding='iso-8859-1')
        df = pd.read_csv(path_complete_data, encoding='iso-8859-1')
        df.drop(df.columns[[0]], axis=1, inplace=True)
        return df

    @staticmethod
    # Unzipping the CSV File
    def unzipfile(zipped):
        zip_ref = zipfile.ZipFile(zipped, 'r')
        zip_ref.extractall(os.path.dirname(os.path.realpath(zip_file_path)))
        zip_ref.close()

    @staticmethod
    # The Main Data Extract Function
    # Run to Extract Data (invokes all the other functions above)
    def obtainDataFromLocalDBs():
        f = FloatProgress(min=0, max=100)
        display(f)
        # unzip the dataset
        DataPreparation.unzipfile(zip_file_path)
        # extract the labels dataframe from the csv files
        lis = []
        for year in rellevant_years_for_labels:
            path = os.path.join(input_labels + '_' + year + '.csv')
            df = DataPreparation.getDataFrameForLabelCSV(path, year)
            lis.append(df)
        labels_df = pd.concat(lis)
        f.value += 10
        # extract all the data dataframe from the csv files
        DataPreparation.cleanCommasFromCSV()
        f.value += 20
        df = DataPreparation.extractDataFromCSV()
        f.value += 20

        # merge (by inner join) the data with the labels
        DataPreparation.mergeDataWithLabels(df, labels_df)
        f.value += 50


class MapVisualizations:
    @staticmethod
    def plotDataOnMap(data, year='mean', feature="Happy Planet Index", binary=False, descripton=''):
        if binary:
            num_colors = 2
        else:
            num_colors = 9
        cols = ['country', feature]
        splitted = feature.split()
        title = feature + ' rate per country'
        imgfile = os.path.join(globe_plots, feature + '_' + year + '.png')
        if descripton == '':
            descripton = '''
            Expected values of the {} rate of countriers. Countries without data are shown in grey.
            Data: World Bank - worldbank.org • Lables: HappyPlanetIndex - happyplanetindex.org'''.format(feature)

        gc = GeonamesCache()
        iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())
        df = pd.read_csv(countries_codes, skiprows=0, usecols=[0, 1], encoding='iso-8859-1')
        data_map = pd.merge(df, data, how='inner', on=['country'])
        if not binary:
            if year == 'mean':
                data_map = data_map[['Country Code', 'country', feature]]
                data_map = data_map.groupby(['Country Code'], sort=False).mean()
            else:
                data_map = data_map[['Country Code', 'year', 'country', feature]]
                data_map = data_map.loc[data_map['year'] == float(year)]
                data_map = data_map[['Country Code', 'country', feature]]
                data_map = data_map.groupby(['Country Code'], sort=False).first()
        data_map.reset_index(inplace=True)
        values = data_map[feature]
        data_map.set_index('Country Code', inplace=True)
        if not binary:
            cm = plt.get_cmap('Greens')
            scheme = [cm(i / num_colors) for i in range(num_colors)]
        else:
            cm = plt.get_cmap('prism')
            scheme = [cm(i * 20 / num_colors) for i in range(num_colors)]
        bins = np.linspace(values.min(), values.max(), num_colors)
        data_map['bin'] = np.digitize(values, bins) - 1
        data_map.sort_values('bin', ascending=False).head(10)
        fig = plt.figure(figsize=(22, 12))

        ax = fig.add_subplot(111, axisbg='w', frame_on=False)
        if not binary:
            if year == 'mean':
                fig.suptitle('mean {} rate for all data'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)
            else:
                fig.suptitle('{} rate in year {}'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)
        else:
            fig.suptitle('{} rate'.format(' '.join(splitted[:7]), year), fontsize=30, y=.95)

        m = Basemap(lon_0=0, projection='robin')
        m.drawmapboundary(color='w')

        f = FloatProgress(min=0, max=100)
        display(f)

        m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
        for info, shape in zip(m.units_info, m.units):
            iso3 = info['ADM0_A3']
            if iso3 not in data_map.index:
                color = '#dddddd'
            else:
                ind = data_map.ix[iso3, 'bin'].astype(np.int64)
                color = scheme[ind]

            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches)
            pc.set_facecolor(color)
            ax.add_collection(pc)
            f.value += 75 / len(m.units_info)

        # Cover up Antarctica so legend can be placed over it.
        ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)

        # Draw color legend.
        ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=3)
        cmap = mpl.colors.ListedColormap(scheme)
        if binary:
            grads = np.linspace(0., 10)
            cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, boundaries=grads, ticks=[0, 10],
                                           orientation='horizontal')
            cb.ax.set_xticklabels(['negative', 'positive'])
        else:
            cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, boundaries=bins, ticks=bins, orientation='horizontal')
            cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])
        f.value += 5

        # Set the map footer.
        plt.annotate(descripton, xy=(-.8, -3.2), size=14, xycoords='axes fraction')
        plt.savefig(imgfile, bbox_inches='tight', pad_inches=.2)
        plt.plot()
        f.value += 20

    @staticmethod
    def plotUncorrolatedCountries(im1, im2, output):
        img1 = cv2.imread(im1, 1)
        img2 = cv2.imread(im2, 1)
        null_img = cv2.imread(template_image, 1)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        null_img = cv2.cvtColor(null_img, cv2.COLOR_BGR2GRAY)

        height1, width1 = img1.shape
        height2, width2 = img2.shape
        height3, width3 = null_img.shape

        min_h = min(height1, height2, height3)
        min_w = min(width1, width2, width3)

        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        null_img = null_img[:min_h, :min_w]

        crop_img = cv2.subtract(img1, img2)[65:900, :]

        null_img = null_img[65:900, :]
        thresh = (255 - crop_img)

        cv2.addWeighted(thresh, 0.5, null_img, 0.5, 0, thresh)
        (threshold, thresh) = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        flag = cv2.imwrite(output, thresh)
        plt.axis('off')
        plt.imshow(thresh, cmap='gray', interpolation='bicubic'), plt.show()


class DataVisualizations:
    @staticmethod
    def twoDimPCAandClustering(factors, show_plots):
        # Initialize the model with 2 parameters -- number of clusters and random state.
        kmeans_model = KMeans(n_clusters=5, random_state=1)
        # Get only the numeric columns from games.
        # Fit the model using the good columns.
        kmeans_model.fit(factors)
        # Get the cluster assignments.
        labels = kmeans_model.labels_
        # Import the PCA model.

        # Create a PCA model.
        pca_2 = PCA(2)
        # Fit the PCA model on the numeric columns from earlier.
        plot_columns = pca_2.fit_transform(factors)
        if show_plots:
            # Make a scatter plot of each game, shaded according to cluster assignment.
            plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
            # Show the plot.
            plt.show()
        return plot_columns, labels

    @staticmethod
    def simple2Dgraph(x_axis, title, xlabel, ylabel, ylim_start, ylim_end, ys, definitions, colors):
        for y, c, defi in zip(ys, colors, definitions):
            lines = plt.plot(x_axis.tolist(), y.tolist(), color=c, label=defi)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylim(ylim_start, ylim_end)
        plt.legend()
        plt.show()
    @staticmethod
    def distPlot(x_axis, title, xlabel, ylabel, bins, kde):
        sns.distplot(x_axis, bins = bins, kde = kde)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


class ImagesUtils:
    @staticmethod
    def concat_images(imga, imgb):
        """
        Combines two color image ndarrays side-by-side.
        """
        ha, wa = imga.shape[:2]
        hb, wb = imgb.shape[:2]
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(shape=(max_height, total_width), dtype=np.uint8)
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa + wb] = imgb
        return new_img

    @staticmethod
    def concat_n_images(image_path_list):
        """
        Combines N color images from a list of image paths.
        """
        output = None
        for i, img_path in enumerate(image_path_list):
            img = plt.imread(img_path)[:, :]
            if i == 0:
                output = img
            else:
                output = ImagesUtils.concat_images(output, img)
        return output


class OutliersDetection():
    @staticmethod
    def avg_r2(g_factors, g_class, n_iter):
        sum = 0.0
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(g_factors, g_class, test_size=0.4, random_state=1)
            regr = linear_model.LinearRegression()
            regr.fit(X_test, y_test)
            sum += regr.score(X_test, y_test)
        return sum / (n_iter * (1.0))

    @staticmethod
    def remove_outliers_rlm(train_factors, train_class, train_data, i, show_plots):
        for i in range(i):
            amount = 0
            dropped_rows = np.asarray([])
            print("Stage", i)
            validation_r_squared = OutliersDetection.avg_r2(train_factors, train_class, 100)
            print("validation R^2, %.4f " % (validation_r_squared))
            rob = sklearn.linear_model.HuberRegressor()
            X = np.asarray(train_factors)
            Y = np.asarray(train_class)
            rob.fit(X, Y)
            y_predicted = rob.predict(X)
            # plotting res vs. pred before dropping outliers
            res = [val for val in (Y - y_predicted)]
            y, x = res, rob.predict(X)
            if show_plots:
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(1, 1, 1)  # one row, one column, first plot
                ax.scatter(x, y, c="blue", alpha=.1, s=300)
                ax.set_title("residuals vs. predicted:")
                ax.set_xlabel("predicted")
                ax.set_ylabel("residuals)")
                plt.show()
            # dropping rows
            res = [abs(val) for val in (Y - y_predicted)]
            rresid = list(zip(range(train_factors.shape[0]), res))
            not_sorted = rresid
            rresid.sort(key=lambda tup: tup[1], reverse=True)
            length = len(rresid)
            sd = np.asarray([tup[1] for tup in rresid]).std()
            mean = np.asarray([tup[1] for tup in rresid]).mean()
            deleted_index = [tup[0] for tup in rresid if tup[1] > mean + 2 * sd]
            amount += len(deleted_index)
            #dropped_rows = train_factors.take(deleted_index, axis=0, convert=True, is_copy=True)
            train_factors = train_factors.drop(train_factors.index[deleted_index])
            train_class = train_class.drop(train_class.index[deleted_index])
            train_data = train_data.drop(train_data.index[deleted_index])
            print("%d rows were dropped" % (amount))
            train_factors.reset_index(drop=True, inplace=True)
            train_class.reset_index(drop=True, inplace=True)
            train_data.reset_index(drop=True, inplace=True)
            # res vs. pred after outliers dropping
        print("After final stage")
        X = np.asarray(train_factors)
        Y = np.asarray(train_class)
        validation_r_squared = OutliersDetection.avg_r2(train_factors, train_class, 100)
        print("validation R^2, %.4f " % (validation_r_squared))
        rob = sklearn.linear_model.HuberRegressor()
        rob.fit(X, Y)
        y_predicted = rob.predict(X)
        res = [val for val in (Y - y_predicted)]
        y, x = res, rob.predict(X)
        if show_plots:
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x, y, c="purple", alpha=.1, s=300)
            ax.set_title("residuals vs. predicted: final")
            ax.set_xlabel("predicted")
            ax.set_ylabel("residuals")
            plt.show()
        return train_factors, train_class, train_data


class ResultsMeasurements():
    def __init__(self, loadModel, trainData, testData, trainFactors, testFactors, trainClass, testClass, model, modelName):
        # A dataframe containing Years, GDP Per Capita, Labels, Predictions
        self.model = model
        self.modelName = modelName
        self.trainRelevantData = pd.DataFrame(trainData['GDP per capita (constant 2005 US$)'])
        self.trainRelevantData['GDP'] = trainData['GDP per capita (constant 2005 US$)']
        self.trainRelevantData['year'] = trainData['year']
        self.trainRelevantData['country'] = trainData['country']
        self.trainRelevantData['label'] = pd.DataFrame(trainClass)
        model_file = modelName.replace(" ", "_")

        if not loadModel:
            self.model.fit(trainFactors, trainClass)
            ModelDump.dumpModelToFile(model_file, model)
        else:
            self.model = ModelDump.loadModelFromFile(model_file)
        self.trainRelevantData['prediction'] = self.model.predict(trainFactors)
        self.trainRelevantData.is_copy = False
        self.trainFactors = trainFactors

        self.testRelevantData = pd.DataFrame(testData['GDP per capita (constant 2005 US$)'])
        self.testRelevantData['GDP'] = testData['GDP per capita (constant 2005 US$)']
        self.testRelevantData['year'] = testData['year']
        self.testRelevantData['country'] = testData['country']
        self.testRelevantData['label'] = pd.DataFrame(testClass)
        self.testRelevantData['prediction'] = self.model.predict(testFactors)
        self.testRelevantData.is_copy = False
        self.testFactors = testFactors

    def RsquaredGraph(self, r2_train, r2_test, x_axis):
        DataVisualizations.simple2Dgraph(r2_train[0],
                                         self.modelName + '\n R^2 per ' + x_axis + ', Train (blue) vs. Test(green)',
                                         x_axis,
                                         'R^2', -4, 1, \
                                         [r2_train[1], r2_test[1]], ['R^2 Train', 'R^2 Test'], ['b', 'g'])

    def RsquaredSeriesYear(self, data, x_axis):
        RsquaredSeries = pd.DataFrame([[i, r2_score(data[data[x_axis] == i].label, data[data[x_axis] == i].prediction)] \
                                       for i in data[x_axis].unique()])
        return RsquaredSeries.sort_values(by=0, ascending=1)

    def RsquaredSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        RsquaredSeries = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                        r2_score(sortedData[i].label, sortedData[i].prediction)] for i in
                                       range(len(sortedData))])
        return RsquaredSeries.sort_values(by=0, ascending=1)

    def RSquaredResults(self):
        RSquaredTrain = self.model.score(self.trainFactors, self.trainRelevantData['label'])
        RSquaredTest = self.model.score(self.testFactors, self.testRelevantData['label'])

        print("R^2 for Train data = " + str(RSquaredTrain))
        print("R^2 for Test data = " + str(RSquaredTest))

        # RSquaredTrainYears = self.RsquaredSeriesYear(self.trainRelevantData, 'year')
        # RSquaredTestYears = self.RsquaredSeriesYear(self.testRelevantData, 'year')
        #
        # RSquaredTrainGDPs = self.RsquaredSeriesGDP(self.trainRelevantData, 'GDP')
        # RSquaredTestGDPs = self.RsquaredSeriesGDP(self.testRelevantData, 'GDP')
        #
        # # R^2 per year, per GDP
        # self.RsquaredGraph(RSquaredTrainYears, RSquaredTestYears, 'Year')
        # self.RsquaredGraph(RSquaredTrainGDPs, RSquaredTestGDPs, 'GDP')

    def DistributionNumericCalc(self, predictions):
        return stats.kstest(predictions, 'norm')

    def DistributionGraphicCalc(self, predictions, binsNum, title):
        sns.distplot(predictions, bins=binsNum, kde=True)
        plt.title(self.modelName + '\n Histogram of Happy Planet Index values: ' + title)
        plt.xlabel('HPI')
        plt.ylabel('density')
        plt.show()

    def DistributionResults(self):
        print("KSTEST results, train: " + str(self.DistributionNumericCalc(self.trainRelevantData['prediction'])))
        print("KSTEST results, test : " + str(self.DistributionNumericCalc(self.testRelevantData['prediction'])))

        self.DistributionGraphicCalc(self.trainRelevantData['label'], 30, "train label")
        self.DistributionGraphicCalc(self.trainRelevantData['prediction'], 30, "train prediction")

        self.DistributionGraphicCalc(self.testRelevantData['label'], 30, "test label")
        self.DistributionGraphicCalc(self.testRelevantData['prediction'], 30, "test prediction")

    def MeanPredictionGraph(self, prediction_train, prediction_test, x_axis):
        DataVisualizations.simple2Dgraph(prediction_train[0],
                                         self.modelName + '\n HPI per ' + x_axis + ', Train Prediction mean (blue) vs. Train Label mean (green)',
                                         x_axis, 'Prediction', 0, 100, \
                                         [prediction_train[1], prediction_train[2]],
                                         ['Train Prediction', 'Train Class'], ['b', 'g'])
        DataVisualizations.simple2Dgraph(prediction_test[0],
                                         self.modelName + '\n HPI per ' + x_axis + ', Test Prediction mean (blue) vs. Test Label mean (green)',
                                         x_axis, 'Prediction', 0, 100, \
                                         [prediction_test[1], prediction_test[2]], ['Test Prediction', 'Test Class'],
                                         ['b', 'g'])

    def MeanPredictionSeriesYear(self, data, x_axis):
        MeanPredictionSeries = pd.DataFrame(
            [[i, data[data[x_axis] == i].prediction.mean(), data[data[x_axis] == i].label.mean()] \
             for i in data[x_axis].unique()])
        return MeanPredictionSeries.sort_values(by=0, ascending=1)

    def MeanPredictionSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        MeanPredictionSeries = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                              sortedData[i].prediction.mean(), sortedData[i].label.mean()] for i in
                                             range(len(sortedData))])
        return MeanPredictionSeries.sort_values(by=0, ascending=1)

    def MeanPredictionResults(self):
        print("The mean HPI of the train data: " + str(self.trainRelevantData['label'].mean()))
        print("The mean prediction of the train data: " + str(self.trainRelevantData['prediction'].mean()))
        print("The mean HPI of the test data : " + str(self.testRelevantData['label'].mean()))
        print("The mean prediction of the test data : " + str(self.testRelevantData['prediction'].mean()))

        MeanPredictionTrainYears = self.MeanPredictionSeriesYear(self.trainRelevantData, 'year')
        MeanPredictionTestYears = self.MeanPredictionSeriesYear(self.testRelevantData, 'year')

        MeanPredictionTrainGDPs = self.MeanPredictionSeriesGDP(self.trainRelevantData, 'GDP')
        MeanPredictionTestGDPs = self.MeanPredictionSeriesGDP(self.testRelevantData, 'GDP')

        # Mean Prediction per year, per GDP
        self.MeanPredictionGraph(MeanPredictionTrainYears, MeanPredictionTestYears, 'Year')
        self.MeanPredictionGraph(MeanPredictionTrainGDPs, MeanPredictionTestGDPs, 'GDP')

    def errPercentage(self, label, prediction):
        return (abs(label - prediction) / label) * 100

    def errPercentageCalc(self, label, prediction):
        errTable = pd.DataFrame({'label': label, 'prediction': prediction})
        errTable['errPercentage'] = errTable.apply(lambda row: self.errPercentage(row['label'], row['prediction']),
                                                   axis=1)
        return errTable['errPercentage'].mean()

    def ErrorPercentageGraph(self, errPer_train, errPer_test, x_axis):
        DataVisualizations.simple2Dgraph(errPer_train[0],
                                         self.modelName + '\n Error Percentage per ' + x_axis + ', Train vs. Test',
                                         x_axis, 'Error Percentage', 0, 100, \
                                         [errPer_train[1], errPer_test[1]],
                                         ['Error Percentage Train', 'Error Percentage Test'], ['b', 'g'])

    def ErrorPercentageSeriesYear(self, data, x_axis):
        ErrorPercentage = pd.DataFrame(
            [[i, self.errPercentageCalc(data[data[x_axis] == i].label, data[data[x_axis] == i].prediction)] \
             for i in data[x_axis].unique()])
        return ErrorPercentage.sort_values(by=0, ascending=1)

    def ErrorPercentageSeriesGDP(self, data, x_axis):
        sortedData = data.sort_values(by='GDP', ascending=1)
        sortedData = np.array_split(sortedData, 30)
        ErrorPercentage = pd.DataFrame([[sortedData[i].iloc[[0]]['GDP'].item(), \
                                         self.errPercentageCalc(sortedData[i].label, sortedData[i].prediction)] for i in
                                        range(len(sortedData))])
        return ErrorPercentage.sort_values(by=0, ascending=1)

    def ErrorPercentageResults(self):
        ErrorPercentageTrain = self.errPercentageCalc(self.trainRelevantData['prediction'],
                                                      self.trainRelevantData['label'])
        ErrorPercentageTest = self.errPercentageCalc(self.testRelevantData['prediction'],
                                                     self.testRelevantData['label'])

        print("Error Percentage for Train data = " + str(ErrorPercentageTrain))
        print("Error Percentage for Test data = " + str(ErrorPercentageTest))

        ErrorPercentageTrainYears = self.ErrorPercentageSeriesYear(self.trainRelevantData, 'year')
        ErrorPercentageTestYears = self.ErrorPercentageSeriesYear(self.testRelevantData, 'year')

        ErrorPercentageTrainGDPs = self.ErrorPercentageSeriesGDP(self.trainRelevantData, 'GDP')
        ErrorPercentageTestGDPs = self.ErrorPercentageSeriesGDP(self.testRelevantData, 'GDP')

        # R^2 per year, per GDP
        self.ErrorPercentageGraph(ErrorPercentageTrainYears, ErrorPercentageTestYears, 'Year')
        self.ErrorPercentageGraph(ErrorPercentageTrainGDPs, ErrorPercentageTestGDPs, 'GDP')


class ModelDump():
    @staticmethod
    def dumpModelToFile(name, model):
        with open(os.path.join(dumped_models, name + '.pkl'), 'wb') as fid:
            cPickle.dump(model, fid)

    @staticmethod
    def loadModelFromFile(name):
        with open(os.path.join(dumped_models, name + '.pkl'), 'rb') as fid:
            model = cPickle.load(fid)
        return model
