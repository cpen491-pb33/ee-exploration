import ee


def classifying():
    ee.Initialize()

    # point = ee.Geometry.Point([-123.116226, 49.246292])
    geo_area = ee.Geometry.BBox(-123.116227, 49.246291, -123.116225, 49.246293)

    # It's a real time image collection and updates to the current date
    imageCollection = 'LANDSAT/LC08/C01/T1_RT'
    # dates = ['2016-01-01', '2016-12-31']
    dates = ['2020-01-01', '2020-12-31']
    min, max, bands = 0, 30000, ['B4', 'B3', 'B2']

    image = ee.ImageCollection(imageCollection) \
        .filterBounds(geo_area) \
        .filterDate(dates[0], dates[1]) \
        .sort('CLOUD_COVER') \
        .first() \
        .select('B[1-7]')
    # .filterBounds(geo_area) \

    vis_params = {
        'min': min,
        'max': max,
        'bands': bands
    }

    imageDate = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print('imageDate', imageDate)
    imageInfo = image.get('CLOUD_COVER').getInfo()
    print('imageInfo', imageInfo)

    # Add labelling
    nlcd = ee.Image('USGS/NLCD/NLCD2016').select('landcover').clip(image.geometry())

    # Make the training dataset.
    points = nlcd.sample(**{
        'region': image.geometry(),
        'scale': 30,
        'numPixels': 5000,
        'seed': 0,
        'geometries': True  # Set this to False to ignore geometries
    })

    print('points size', points.size().getInfo())
    print('points info', points.first().getInfo())

    # Use these bands for prediction.
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    # This property of the table stores the land cover labels.
    label = 'landcover'

    # Overlay the points on the imagery to get training.
    training = image.select(bands).sampleRegions(**{
        'collection': points,
        'properties': [label],
        'scale': 30
    })

    # Train a CART classifier with default parameters.
    trained = ee.Classifier.smileCart().train(training, label, bands)

    print('training data info', training.first().getInfo())

    # Classify the image with the same bands used for training.
    result = image.select(bands).classify(trained)

    class_palette = nlcd.get('landcover_class_palette').getInfo()
    print('class palette', class_palette)
    class_values = nlcd.get('landcover_class_values').getInfo()
    print('class values', class_values)
    class_names = nlcd.get('landcover_class_names').getInfo()
    for i, name in enumerate(class_names):
        class_names[i] = name.split('-')[0].strip()
    print('class names', class_names)

    landcover = result.set('classification_class_values', class_values)
    landcover = landcover.set('classification_class_palette', class_palette)

    task = ee.batch.Export.image.toDrive(**{
        'image': landcover,
        'description': 'landCoverExample',
        'folder': 'Image_Folder',
        'scale': 30,
        'region': geo_area.getInfo()['coordinates']
    })
    task.start()


if __name__ == '__main__':
    classifying()
