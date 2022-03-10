#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import cv2
import os
import sys
#
import ulises_uclick as ulises
#
def ClickHorizon(pathBasis, overwrite, verbosePlot):
    #
    for root, _, filenames in sorted(os.walk(pathBasis)):
        for filename in sorted([item for item in filenames if item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']]):
            #
            print('... clicking horizon at {:}'.format(filename))
            if input('Continue (n/[y])? ') in ['n', 'N']:
                print('stop clicking horizon') 
                sys.exit()
            #
            img = cv2.imread(root + os.sep + filename)
            #
            pathCdhTxt = root + os.sep + filename[0:filename.rfind('.')] + 'cdh.txt'
            ulises.ClickCdhTxt(img, pathCdhTxt, options={'overwrite':overwrite, 'titlePreamble':'{:}: '.format(filename)})
            #
            if verbosePlot:
                chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
                if len(chs) == len(rhs) == 0:
                    continue
                pathFolderOut = pathBasis + os.sep + '..' + os.sep + 'TMP' # IMP*
                ulises.MakeFolder(pathFolderOut)
                ulises.PlotCRWithTextsInImage(img, chs, rhs, pathFolderOut + os.sep + filename[0:-4] + 'cdh_check.jpg', options={'texts':None})
    #
    return None
#
def ClickGCPs(pathMain, pathBasis, overwrite, verbosePlot):
    for root, _, filenames in sorted(os.walk(pathBasis)):
        for filename in sorted([item for item in filenames if item[item.rfind('.')+1:] in ['jpeg', 'jpg', 'png']]):
            #
            print('... clicking GCPs at {:}'.format(filename))
            if input('Continue (n/[y])? ') in ['n', 'N']:
                print('stop clicking GCPs') 
                sys.exit()
            #
            img = cv2.imread(root + os.sep + filename)
            #
            pathGCPsTxt = pathMain + os.sep + 'GCPs.txt'
            pathCdgTxt = root + os.sep + filename[0:filename.rfind('.')] + 'cdg.txt'
            ulises.ClickCdgTxt(img, pathGCPsTxt, pathCdgTxt, options={'overwrite':overwrite, 'titlePreamble':'{:}: '.format(filename)})
            #
            if verbosePlot:
                cs, rs, _, _, _, codes = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':True, 'readOnlyGood':True})
                if len(cs) == len(rs) == 0:
                    continue
                pathFolderOut = pathBasis + os.sep + '..' + os.sep + 'TMP' # IMP*
                ulises.MakeFolder(pathFolderOut)
                ulises.PlotCRWithTextsInImage(img, cs, rs, pathFolderOut + os.sep + filename[0:-4] + 'cdg_check.jpg', options={'texts':codes})
    return None
#
def CheckGCPs(pathBasisCheck, errorCritical):
    #
    eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax = 0.8, 0.999999, errorCritical, 50000
    #
    # check GCPs
    fnsImages = sorted([item for item in os.listdir(pathBasisCheck) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        print('... checking of {:}'.format(fnImage))
        #
        # load image information and dataBasic
        nr, nc = cv2.imread(pathBasisCheck + os.sep + fnImage).shape[0:2]
        oca, ora = (nc-1)/2, (nr-1)/2
        #
        # load GCPs
        pathCdgTxt = pathBasisCheck + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        possGood = ulises.RANSACForGCPs(cs, rs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={'nOfK1asa2':1000})[0]
        #
        # inform
        if possGood is None:
            print('... too few GCPs to be checked')
        elif len(possGood) < len(cs):
            print('... re-run or consider to ignore the following GCPs')
            for pos in [item for item in range(len(cs)) if item not in possGood]:
                c, r, x, y, z = [item[pos] for item in [cs, rs, xs, ys, zs]]
                print('... c = {:8.2f}, r = {:8.2f}, x = {:8.2f}, y = {:8.2f}, z = {:8.2f}'.format(c, r, x, y, z))
        else:
            print('... all the GCPs for {:} are OK'.format(fnImage))
    #
    return None
#
