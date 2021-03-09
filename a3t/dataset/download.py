"""
Modified from https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
"""
from urllib.request import urlopen
import sys
import os
import zipfile


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urlopen(url)
    try:
        f = open(filepath, 'wb')
    except:
        print("Cannot write %s" % filepath)
        raise Exception
    try:
        filesize = int(u.info().get_all("Content-Length")[0])
    except:
        print("URL %s failed to report length" % url)
        filesize = None
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        if filesize is not None:
            status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
            print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_wordvecs(dirpath, all_vocab, dim):
    if all_vocab == 6:
        url = 'http://www-nlp.stanford.edu/data/glove.6B.zip'
    else:
        url = 'http://www-nlp.stanford.edu/data/glove.%dB.%dd.zip' % (all_vocab, dim)
    unzip(download(url, dirpath))


def download_ppdb(dirpath, ppdb_type):
    download("http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-%s-lexical" % ppdb_type, dirpath)


def download_artifacts(dataset_home):
    a3t_artifact_file = os.path.join(dataset_home, "A3T-artifacts", "dataset")
    a3t_enkey_file = os.path.join(a3t_artifact_file, "en.key")
    a3t_sst2test_file = os.path.join(a3t_artifact_file, "sst2test.txt")
    if not os.path.exists(a3t_enkey_file) or not os.path.exists(a3t_sst2test_file):
        unzip(download("https://github.com/ForeverZyh/A3T/archive/artifacts.zip", os.path.join("/tmp", ".A3T")))
