from optparse import OptionParser

parser = OptionParser(usage="usage: %prog datapath [options]")

parser.add_option("-d", dest='dataset', type=str, metavar='datasetpath', help=f'path to the pickled dataset')

parser.add_option("-o", "--output", dest="output",
                  help="Result file", type=str,  default='../log/multiModal_log.csv')

parser.add_option("-X", "--posteriors", dest="posteriors", action='store_true',
                  help="Add posterior probabilities to the document embedding representation", default=False)

parser.add_option("-W", "--supervised", dest="supervised", action='store_true',
                  help="Add supervised (Word-Class Embeddings) to the document embedding representation", default=False)

parser.add_option("-M", "--pretrained", dest="pretrained", action='store_true',
                  help="Add pretrained MUSE embeddings to the document embedding representation", default=False)

parser.add_option("-B", "--mbert", dest="mbert", action='store_true',
                  help="Add multilingual Bert (mBert) document embedding representation", default=False)

parser.add_option('-G', dest='gruViewGenerator',  action='store_true',
                  help="Add document embedding generated via recurrent net (GRU)", default=False)

parser.add_option("--l2", dest="l2", action='store_true',
                  help="Activates l2 normalization as a post-processing for the document embedding views",
                  default=True)

parser.add_option("--allprob", dest="allprob", action='store_true',
                  help="All views are generated as posterior probabilities. This affects the supervised and pretrained"
                       "embeddings, for which a calibrated classifier is generated, which generates the posteriors",
                  default=False)

parser.add_option("--feat-weight", dest="feat_weight",
                  help="Term weighting function to weight the averaged embeddings", type=str,  default='tfidf')

parser.add_option("-w", "--we-path", dest="we_path",
                  help="Path to the MUSE polylingual word embeddings", default='../embeddings')

parser.add_option("-s", "--set_c", dest="set_c", type=float,
                  help="Set the C parameter", default=1)

parser.add_option("-c", "--optimc", dest="optimc", action='store_true',
                  help="Optimize hyperparameters", default=False)

parser.add_option("-j", "--n_jobs", dest="n_jobs", type=int,
                  help="Number of parallel jobs (default is -1, all)", default=-1)

parser.add_option("-p", "--pca", dest="max_labels_S", type=int,
                  help="If smaller than number of target classes, PCA will be applied to supervised matrix. ",
                  default=300)

parser.add_option("-r", "--remove-pc", dest="sif", action='store_true',
                  help="Remove common component when computing dot product of word embedding matrices", default=True)

parser.add_option("-z", "--zscore", dest="zscore", action='store_true',
                  help="Z-score normalize matrices (WCE and MUSE)", default=True)

parser.add_option("-a", "--agg", dest="agg", action='store_true',
                  help="Set aggregation function of the common Z-space to average (Default: concatenation)",
                  default=False)

# ------------------------------------------------------------------------------------

parser.add_option('--hidden', type=int, default=512, metavar='int',
                  help='hidden lstm size (default: 512)')

parser.add_option('--sup-drop', type=float, default=0.5, metavar='[0.0, 1.0]',
                  help='dropout probability for the supervised matrix (default: 0.5)')

parser.add_option('--tunable', action='store_true', default=False,
                  help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')

parser.add_option('--logfile_gru', dest='logfile_gru', default='../log/log_gru_viewgenerator.csv')

parser.add_option('--seed', type=int, default=1, metavar='int', help='random seed (default: 1)')

parser.add_option('--force', action='store_true', default=False,
                  help='do not check if this experiment has already been run')

parser.add_option('--gruMuse', dest='gruMUSE', action='store_true', default=False,
                  help='Deploy MUSE embedding as embedding layer of the GRU View Generator')

parser.add_option('--gruWce', dest='gruWCE', action='store_true', default=False,
                  help='Deploy WCE embedding as embedding layer of the GRU View Generator')

parser.add_option('--gru-path', dest='gru_path', default=None,
                  help='Set the path to a pretrained GRU model (aka, -G view generator)')

parser.add_option('--bert-path', dest='bert_path', default=None,
                  help='Set the path to a pretrained mBERT model (aka, -B view generator)')
