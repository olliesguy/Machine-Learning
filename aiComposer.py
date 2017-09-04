import urllib
import zipfile

url = "www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip"
urllib.urlretrieve.(url, "dataset.zip")

zip = zipfile.ZipFile(r'dataset.zip')
zip.extractall('data')

# Music Language Modelling using recurrent neural networks.

nottingham_util.create_model()

rnn.train_model()
