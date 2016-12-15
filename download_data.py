import os
import zipfile


data_url = "http://lmb.informatik.uni-freiburg.de/\
resources/binaries/eccv_2016_cars/data.zip"
snapshots_url = "http://lmb.informatik.uni-freiburg.de/\
resources/binaries/eccv_2016_cars/snapshots.zip"


def download_and_unpack(url, dest):
    if not os.path.exists(dest):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        print("downloading " + dest + " ...")

        archive_name = dest + ".zip"
        urllib.urlretrieve(url, archive_name)
        in_file = open(archive_name, 'rb')
        z = zipfile.ZipFile(in_file)
        for name in z.namelist():
            print("extracting " + name)
            outpath = "./"
            z.extract(name, outpath)
        in_file.close()
        os.remove(archive_name)

        print("done.")

download_and_unpack(snapshots_url, "snapshots")
download_and_unpack(data_url, "data")
