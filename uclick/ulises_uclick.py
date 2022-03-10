#
# Thu Mar 10 15:17:59 2022, extract from Ulises by Gonzalo Simarro
#
import os
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
#
def AB2Pa11(A, b): # *** 
    '''
    .- input A is a (2*nx11)-float-ndarray
    .- input b is a 2*n-float-ndarray
    .- output Pa11 is a 11-float-ndarray
    '''
    try:
        Pa11 = np.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
    except:
        Pa11 = None
    return Pa11
def AreImgMarginsOK(nc, nr, imgMargins): # 202109101200 # *** 
    ''' comments:
    .- input nc and nr are integers
    .- input imgMargins is a dictionary
    .- output areImgMarginsOK is a boolean
    '''
    imgMargins = CompleteImgMargins(imgMargins)
    condC = min([imgMargins['c0'], imgMargins['c1'], nc-1-(imgMargins['c0']+imgMargins['c1'])]) >= 0
    condR = min([imgMargins['r0'], imgMargins['r1'], nr-1-(imgMargins['r0']+imgMargins['r1'])]) >= 0
    areImgMarginsOK = condC and condR
    return areImgMarginsOK
def CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2): # *** 
    ''' comments:
    .- input cDs and rDs are float-ndarrays
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cUs and rUs are float-ndarrays
    '''
    if np.abs(k1asa2) < 1.e-14:
        cUs = 1. * cDs
        rUs = 1. * rDs
    else:
        dDs2 = (cDs - oca) ** 2 + (rDs - ora) ** 2
        xias = k1asa2 * dDs2
        xas = Xi2XForParabolicDistortion(xias)
        cUs = (cDs - oca) / (1. + xas) + oca
        rUs = (rDs - ora) / (1. + xas) + ora
    cDsR, rDsR = CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2)
    assert np.allclose(cDs, cDsR) and np.allclose(rDs, rDsR)
    return cUs, rUs
def CR2CRInteger(cs, rs): # 202109131000 # *** 
    ''' comments:
    .- input cs and rs are integer- or float-ndarrays
    .- output cs and rs are integer-ndarrays
    '''
    cs = np.round(cs).astype(int)
    rs = np.round(rs).astype(int)
    return cs, rs
def CR2CRIntegerWithinImage(nc, nr, cs, rs, options={}): # 202109141700 # *** 
    ''' comments:
    .- input nc and nr are integers or floats
    .- input cs and rs are float-ndarrays
    .- output csIW and rsIW are integer-ndarrays
    '''
    keys, defaultValues = ['imgMargins'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    csI, rsI = CR2CRInteger(cs, rs)
    optionsTMP = {'imgMargins':imgMargins, 'rounding':False}
    possWithin = CR2PositionsWithinImage(nc, nr, csI, rsI, optionsTMP)
    csIW, rsIW = csI[possWithin], rsI[possWithin]
    return csIW, rsIW
def CR2PositionsWithinImage(nc, nr, cs, rs, options={}): # 202109131400 # *** 
    ''' comments:
    .- input nc and nr are integers
    .- input cs and rs are integer- or float-ndarrays
    .- output possWithin is an integer-list
    '''
    keys, defaultValues = ['imgMargins', 'rounding'], [{'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, False]
    options = CompleteADictionary(options, keys, defaultValues)
    imgMargins = CompleteImgMargins(options['imgMargins'])
    assert AreImgMarginsOK(nc, nr, imgMargins)
    if options['rounding']:
        cs, rs = CR2CRInteger(cs, rs)
    cMin, cMax = imgMargins['c0'], nc-1-imgMargins['c1'] # recall that img[:, nc-1, :] is OK, but not img[:, nc, :]
    rMin, rMax = imgMargins['r0'], nr-1-imgMargins['r1'] # recall that img[nr-1, :, :] is OK, but not img[nr, :, :]
    possWithin = np.where((cs >= cMin) & (cs <= cMax) & (rs >= rMin) & (rs <= rMax))[0]
    return possWithin
def CURU2B(cUs, rUs): # *** 
    poss0, poss1 = Poss0AndPoss1(len(cUs))    
    b = np.zeros(2 * len(cUs))
    b[poss0] = cUs
    b[poss1] = rUs
    return b
def CURU2CDRDForParabolicSquaredDistortion(cUs, rUs, oca, ora, k1asa2): # *** 
    ''' comments:
    .- input cUs and rUs are float-ndarrays
    .- input oca and ora are floats
    .- input k1asa2 is a float (k1a * sa ** 2)
    .- output cUs and rUs are float-ndarrays
    '''
    dUs2 = (cUs - oca) ** 2 + (rUs - ora) ** 2
    cDs = (cUs - oca) * (1. + k1asa2 * dUs2) + oca
    rDs = (rUs - ora) * (1. + k1asa2 * dUs2) + ora
    return cDs, rDs
def CURUXYZ2A(cUs, rUs, xs, ys, zs): # 202201250813
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A is a (2*len(cUs)x11)-float-ndarray
    '''
    A0 = XYZ2A0(xs, ys, zs)
    A1 = CURUXYZ2A1(cUs, rUs, xs, ys, zs)
    A = np.concatenate((A0, A1), axis=1)
    return A
def CURUXYZ2A1(cUs, rUs, xs, ys, zs): # 202201250812
    '''
    .- input cUs, rUs, xs, ys and zs are float-ndarrays of the same length
    .- output A1 is a (2*len(cUs)x3)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(cUs))
    A1 = np.zeros((2 * len(xs), 3))
    A1[poss0, 0], A1[poss0, 1], A1[poss0, 2] = -cUs*xs, -cUs*ys, -cUs*zs
    A1[poss1, 0], A1[poss1, 1], A1[poss1, 2] = -rUs*xs, -rUs*ys, -rUs*zs
    return A1
def ClickAPixelInImage(img, options={}): # 202110061655
    ''' comments:
    .- input img is a cv2-image or a string
    .- output cs and rs are float-lists
    '''
    keys, defaultValues = ['c0', 'r0', 'factor', 'pathImgOut', 'pathTxtOut', 'title'], [0, 0, 0., None, None, '']
    options = CompleteADictionary(options, keys, defaultValues)
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[0:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.isclose(options['factor'], 0.):
        plt.figure()
        plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
    else:
        plt.figure(figsize=(nr*options['factor'], nc*options['factor']))
    plt.title(options['title'], fontsize=16)
    ttl = plt.gca().title
    ttl.set_position([.5, 1.05])
    plt.imshow(img, interpolation='none')
    print('... ClickPixelsInImage: please click')
    csrs = plt.ginput(1, timeout=0)
    plt.close('all')
    cs, rs = [item[0]+options['c0'] for item in csrs], [item[1]+options['r0'] for item in csrs]
    if options['pathImgOut'] is not None:
        print('... ClickAPixelInImage: plotting pixels at {:}'.format(options['pathImgOut']))
        img = DisplayCRInImage(img, cs, rs, options={'colors':[[ 0,   0,   0]], 'size':np.sqrt(nc * nr) / 100})
        img = DisplayCRInImage(img, cs, rs, options={'colors':[[ 0, 255, 255]], 'size':np.sqrt(nc * nr) / 200})
        cv2.imwrite(options['pathImgOut'], img)
    if options['pathTxtOut'] is not None:
        print('... ClickPixelsInImage: saving pixels at {:}'.format(options['pathTxtOut']))
        fileout = open(options['pathTxtOut'], 'w')
        for pos in range(len(cs)):
            fileout.write('{:12.3f} {:12.3f} \t c and r \n'.format(cs[pos], rs[pos]))
        fileout.close()
    return cs, rs
def ClickAPixelInImageInTwoSteps(img, options={}): # 202107061100 # *** 
    ''' comments:
    .- input img is a cv2-image or a string
    .- output c and r are floats or Nones
    .- output nOfClicks is an integer (0, 1 or 2)
    '''
    keys, defaultValues = ['factor', 'pathImgOut', 'pathTxtOut', 'sizeOfWindow', 'title'], [0., None, None, 100, '']
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- options['pathFileImgOut'] is a string or None
        .- if pathFileImgOut is None, does not plot the pixels
    .- options['pathFileTxtOut'] is a string or None
        .- if pathFileTxtOut is None, does not write the pixels
    '''
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[0:2]
    optionsTMP = {'c0':0, 'r0':0, 'factor':options['factor'], 'pathImgOut':None, 'pathTxtOut':None, 'title':options['title'] + ' (first approach)'}
    cs, rs = ClickAPixelInImage(img, optionsTMP)
    if len(cs) == len(rs) == 0:
        return None, None, 0
    elif len(cs) == len(rs) == 1:
        c, r = cs[0], rs[0]
    else:
        assert False
    c0, c1 = max([int(c - options['sizeOfWindow']/2), 0]), min([int(c + options['sizeOfWindow']/2), nc])
    r0, r1 = max([int(r - options['sizeOfWindow']/2), 0]), min([int(r + options['sizeOfWindow']/2), nr])
    imgZoom = img[r0:r1, c0:c1, :]
    optionsTMP = {'c0':c0, 'r0':r0, 'factor':5*options['factor'], 'pathImgOut':None, 'pathTxtOut':None, 'title':options['title']}
    cs, rs = ClickAPixelInImage(imgZoom, optionsTMP)
    if len(cs) == len(rs) == 0:
        return None, None, 0
    elif len(cs) == len(rs) == 1:
        c, r, nOfClicks = cs[0], rs[0], 1
    else:
        assert False
    try:
        if options['pathImgOut'] is not None:
            print('... ClickPixelInTwoStepsInImage: plotting pixels at {:}'.format(options['pathImgOut']))
            img = DisplayCRInImage(img, np.asarray([c]), np.asarray([r]), options={'colors':[[ 0,   0,   0]], 'size':np.sqrt(nc * nr) / 100})
            img = DisplayCRInImage(img, np.asarray([c]), np.asarray([r]), options={'colors':[[ 0, 255, 255]], 'size':np.sqrt(nc * nr) / 200})
            cv2.imwrite(options['pathImgOut'], img)
    except:
        print('*** ClickPixelInTwoStepsInImage: error plotting pixels')
    try:
        if options['pathTxtOut'] is not None:
            print('... ClickPixelInTwoStepsInImage: saving pixels at {:}'.format(options['pathTxtOut']))
            fileout = open(options['pathTxtOut'], 'w')
            fileout.write('{:12.3f} {:12.3f} \t c and r \n'.format(c, r))
            fileout.close()
    except:
        print('*** ClickPixelInTwoStepsInImage: error saving pixels')
    return c, r, nOfClicks
def ClickCdgTxt(img, pathCdeTxt, pathCdgTxt, options={}): # 202110050942
    ''' comments:
    .- input img is a cv2-image or a string
    .- input pathCdeTxt is a string
    .- input pathCdgTxt is a string
    '''
    keys, defaultValues = ['factor', 'overwrite', 'sizeOfWindow', 'titlePreamble'], [0., False, 100, '']
    options = CompleteADictionary(options, keys, defaultValues)
    img = PathImgOrImg2Img(img)
    codes, xsForCodes, ysForCodes, zsForCodes, _, areCodesOnForCodes = ReadCdeTxt(pathCdeTxt)
    if os.path.exists(pathCdgTxt):
        csCdg, rsCdg, xsCdg, ysCdg, zsCdg = ReadCdgTxt(pathCdgTxt, {'readCodes':False, 'readOnlyGood':False})[0:5]
    cs, rs, xs, ys, zs = [{} for item in range(5)] # dictionaries for codes (could be lists, including another list for codes)
    for theCode in [item for item in sorted(codes) if areCodesOnForCodes[item]]: # IMP* sorted
        posInCdg = None
        if os.path.exists(pathCdgTxt) and len(csCdg) > 0: # look if this theCode is already in pathCdgTxt
            ds = np.sqrt((xsCdg - xsForCodes[theCode]) ** 2 + (ysCdg - ysForCodes[theCode]) ** 2 + (zsCdg - zsForCodes[theCode]) ** 2)
            if np.min(ds) < 1.e-3: # IMP*
                posInCdg = np.argmin(ds) # already in pathCdgTxt, and posInCdg is its position ndarrays csCdg, rsCdg, ...
        if posInCdg is None or options['overwrite']: # need to click
            optionsTMP = {'factor':options['factor'], 'sizeOfWindow':options['sizeOfWindow'], 'title':'{:}click point {:}'.format(options['titlePreamble'], theCode)}
            c, r, nOfClicks = ClickAPixelInImageInTwoSteps(img, options=optionsTMP)
            if c is None or r is None or nOfClicks != 1:
                c, r = -999, -999 # IMP* WATCH OUT
            cs[theCode], rs[theCode], xs[theCode], ys[theCode], zs[theCode] = c, r, xsForCodes[theCode], ysForCodes[theCode], zsForCodes[theCode]
        else: # already in pathCdgTxt
            cs[theCode], rs[theCode], xs[theCode], ys[theCode], zs[theCode] = [item[posInCdg] for item in [csCdg, rsCdg, xsCdg, ysCdg, zsCdg]]
    (cs, rs, xs, ys, zs), codes = [list(item.values()) for item in [cs, rs, xs, ys, zs]], list(cs.keys())
    WriteCdgTxt(pathCdgTxt, cs, rs, xs, ys, zs, options={'codes':codes})
    return None
def ClickCdhTxt(img, pathCdhTxt, options={}): # 202110150935
    ''' comments:
    .- input img is a cv2-image or a string
    .- input pathCdhTxt is a string
    '''
    keys, defaultValues = ['factor', 'overwrite', 'sizeOfWindow', 'titlePreamble'], [0., False, 100, '']
    options = CompleteADictionary(options, keys, defaultValues)
    if os.path.exists(pathCdhTxt) and not options['overwrite']:
        return None
    img = PathImgOrImg2Img(img)
    chs, rhs = [], []
    while True:
        optionsTMP = {'factor':options['factor'], 'sizeOfWindow':options['sizeOfWindow'], 'title':'{:}click a point of the horizon'.format(options['titlePreamble'])}
        ch, rh, nOfClicks = ClickAPixelInImageInTwoSteps(img, options=optionsTMP)
        if ch is None or rh is None or nOfClicks != 1:
            break
        chs.append(ch); rhs.append(rh)
    WriteCdhTxt(pathCdhTxt, chs, rhs)
    return None
def CompleteADictionary(theDictionary, keys, defaultValues): # 202109101200 # *** 
    ''' comments:
    .- input theDictionary is a dictionary
    .- input keys is a string-list
    .- input defaultValues is a list of the same length of keys or a single value (string, float, integer or None)
    .- output theDictionary is a dictionary that includes keys and defaultValues for the keys not in input theDictionary
    '''
    if set(keys) <= set(theDictionary.keys()): # no work to do
        pass
    else:
        if isinstance(defaultValues, (list)): # defaultValues is a list
            assert len(keys) == len(defaultValues)
            for posKey, key in enumerate(keys):
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues[posKey]
        else: # defaultValues is a single value
            for key in keys:
                if key not in theDictionary.keys(): # only assigns if there is no key
                    theDictionary[key] = defaultValues
    return theDictionary
def CompleteImgMargins(imgMargins): # 202109101200 # *** 
    ''' comments:
    .- input imgMargins is a dictionary or None
        .- if imgMargins['isComplete'], then it does nothing
        .- if imgMargins is None, then it is initialized to {'c':0, 'r':0}
        .- if imgMargins includes 'c', then generates 'c0' and 'c1' (if not included); otherwise, 'c0' and 'c1' must already be included
        .- if imgMargins includes 'r', then generates 'r0' and 'r1' (if not included); otherwise, 'r0' and 'r1' must already be included
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    .- output imgMargins is a dictionary (including at least 'c0', 'c1', 'r0' and 'r1' and 'isComplete')
        .- imgMargins['c*'] and imgMargins['r*'] are integers
    '''
    if imgMargins is not None and 'isComplete' in imgMargins.keys() and imgMargins['isComplete']:
        return imgMargins
    if imgMargins is None:
        imgMargins = {'c':0, 'r':0}
    for letter in ['c', 'r']:
        try:
            assert int(imgMargins[letter]) == imgMargins[letter]
        except: # imgMargins[letter] is not an integer (it is None or it even does not exist)
            for number in ['0', '1']: # check that c0(r0) and c1(r1) are already in imgMargins
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            continue # go to the next letter since letter+number already ok for this letter
        for number in ['0', '1']:
            try: 
                assert int(imgMargins[letter+number]) == imgMargins[letter+number]
            except:
                imgMargins[letter+number] = imgMargins[letter]
    imgMargins['isComplete'] = True
    return imgMargins
def DisplayCRInImage(img, cs, rs, options={}): # 202109141700 # *** 
    ''' comments:
    .- input img is a cv2 image
    .- input cs and rs are integer- or float-ndarrays of the same length
        .- they are not required to be within the image
    .- output imgOut is a cv2 image
    '''
    keys, defaultValues = ['colors', 'imgMargins', 'size'], [[[0, 0, 0]], {'c0':0, 'c1':0, 'r0':0, 'r1':0, 'isComplete':True}, 2]
    options = CompleteADictionary(options, keys, defaultValues)
    ''' comments:
    .- options['colors'] is a list of colors:
        .- if len(options['colors']) = 1, all the pixels have the same color
        .- if len(options['colors']) > 1, it must be len(options['colors']) > len(cs)
    '''
    imgOut = copy.deepcopy(img)
    nr, nc = img.shape[0:2]
    optionsTMP = {item:options[item] for item in ['imgMargins']}
    csIW, rsIW = CR2CRIntegerWithinImage(nc, nr, cs, rs, optionsTMP)
    if len(options['colors']) == 1:
        colors = [options['colors'][0] for item in range(len(csIW))]
    else: # we do not require
        assert len(options['colors']) >= len(csIW) == len(rsIW)
        colors = options['colors']
    if len(csIW) == len(rsIW) > 0:
        for pos in range(len(csIW)):
            cv2.circle(imgOut, (csIW[pos], rsIW[pos]), int(options['size']), colors[pos], -1)
    return imgOut
def GCPs2K1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2Min, k1asa2Max, options={}): # *** 
    ''' comments:
    .- input cDs, rDs, xs, ys and zs are is a float-ndarrays of the same length
    .- input oca, ora, k1asa2Min, k1asa2Max are floats
    .- output k1asa2 is a float
    '''
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    A0, k1asa2s, errors = XYZ2A0(xs, ys, zs), np.linspace(k1asa2Min, k1asa2Max, options['nOfK1asa2']), []
    for k1asa2 in k1asa2s:
        cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
        A, b = np.concatenate((A0, CURUXYZ2A1(cUs, rUs, xs, ys, zs)), axis=1), CURU2B(cUs, rUs)
        Pa11 = AB2Pa11(A, b)
        cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11)
        errors.append(np.sqrt(np.mean((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2)))
    k1asa2 = k1asa2s[np.argmin(np.asarray(errors))]
    return k1asa2
def MakeFolder(pathFolder): # 202109131100 # *** 
    ''' comments:
    .- input pathFolder is a string
    .- creates a folder if pathFolder does not exist
    '''
    if not os.path.exists(pathFolder):
        os.makedirs(pathFolder)
    return None
def NForRANSAC(eRANSAC, pRANSAC, sRANSAC): # *** 
    ''' comments:
    .- input eRANSAC is a float (probability of a point being "bad")
    .- input pRANSAC is a float (goal probability of sRANSAC points being "good")
    .- input sRANSAC is an integer (number of points of the model)
    .- note that: (1 - e) ** s is the probability of a set of s points being good
    .- note that: 1 - (1 - e) ** s is the probability of a set of s points being bad (at least one is bad)
    .- note that: (1 - (1 - e) ** s) ** N is the probability of choosing N sets all being bad
    .- note that: 1 - (1 - (1 - e) ** s) ** N is the probability of choosing N sets where at least one set if good
    .- note that: from 1 - (1 - (1 - e) ** s) ** N = p -> 1 - p = (1 - (1 - e) ** s) ** N and we get the expression
    '''
    N = int(np.log(1. - pRANSAC) / np.log(1. - (1. - eRANSAC) ** sRANSAC)) + 1
    return N
def PathImgOrImg2Img(img): # 202110041642
    ''' comments:
    .- input img is a cv2-image or a string
    .- output img is a cv2-image
    '''
    try:
        nr, nc = img.shape[0:2]
    except:
        img = cv2.imread(img)
        nr, nc = img.shape[0:2]
    img[nr-1, nc-1, 0]
    return img
def PlotCRWithTextsInImage(img, cs, rs, pathOut, options={}): # 202110051044
    ''' comments:
    .- input img is a cv2-image or a string
    .- input cs and rs are float-ndarrays
    .- pathOut is a string
    '''
    keys, defaultValues = ['texts'], None
    options = CompleteADictionary(options, keys, defaultValues)
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[0:2]
    img = DisplayCRInImage(img, cs, rs, {'colors':[[0, 0, 0]], 'size':np.sqrt(nc*nr)/2.e+2+1})
    img = DisplayCRInImage(img, cs, rs, {'colors':[[0, 255, 255]], 'size':np.sqrt(nc*nr)/4.e+2+1})
    if options['texts'] is not None:
        for pos in range(len(cs)):
            cv2.putText(img, options['texts'][pos], (min([int(cs[pos]), int(0.92*nc)]), int(rs[pos])), cv2.FONT_HERSHEY_SIMPLEX, np.sqrt(nc*nr)/2500., (255, 0, 0), int(np.sqrt(nc*nr)/1500.)+1, cv2.LINE_AA)
    MakeFolder(pathOut[0:pathOut.rfind(os.sep)])
    cv2.imwrite(pathOut, img)
    return None
def Poss0AndPoss1(n): # 202201250804
    '''
    .- input n is an integer
    .- output poss0 and poss1 are n-integer-list
    '''
    poss0 = [2*pos+0 for pos in range(n)]
    poss1 = [2*pos+1 for pos in range(n)]
    return poss0, poss1
def RANSACForGCPs(cDs, rDs, xs, ys, zs, oca, ora, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax, options={}): # *** 
    if len(cDs) < 6:
        return None, None
    keys, defaultValues = ['nOfK1asa2'], [1000]
    options = CompleteADictionary(options, keys, defaultValues)
    dD2Max = np.max((cDs - oca) ** 2 + (rDs - ora) ** 2)
    k1asa2 = 0.
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, 3. * ecRANSAC, NForRANSACMax) # WATCH OUT
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]]
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # WATCH OUT
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    possGood = RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax)
    cDsSel, rDsSel, xsSel, ysSel, zsSel = [item[possGood] for item in [cDs, rDs, xs, ys, zs]] # departing from the original points
    k1asa2Min, k1asa2Max = -4./(27.*dD2Max)+1.e-11, 4./(27.*dD2Max) # WATCH OUT
    k1asa2 = GCPs2K1asa2(cDsSel, rDsSel, xsSel, ysSel, zsSel, oca, ora, k1asa2Min, k1asa2Max, options={'nOfK1asa2':options['nOfK1asa2']})
    return possGood, k1asa2
def RANSACForGCPsAndK1asa2(cDs, rDs, xs, ys, zs, oca, ora, k1asa2, eRANSAC, pRANSAC, ecRANSAC, NForRANSACMax): # *** 
    cUs, rUs = CDRD2CURUForParabolicSquaredDistortion(cDs, rDs, oca, ora, k1asa2)
    AAll, bAll = CURUXYZ2A(cUs, rUs, xs, ys, zs), CURU2B(cUs, rUs)
    sRANSAC = 6 #  (to obtain 12 equations >= 11 unkowns)
    N, possGood = min(NForRANSACMax, NForRANSAC(eRANSAC, pRANSAC, sRANSAC)), []
    for iN in range(N):
        possH = random.sample(range(0, len(cUs)), sRANSAC)
        poss01 = [2*item for item in possH] + [2*item+1 for item in possH] # who cares about the order (both A and b suffer the same)
        A, b = AAll[poss01, :], bAll[poss01]
        try:
            Pa11 = AB2Pa11(A, b)
            cUsR, rUsR = XYZPa112CURU(xs, ys, zs, Pa11) # all positions
            errors = np.sqrt((cUsR - cUs) ** 2 + (rUsR - rUs) ** 2)
        except:
            continue
        possGoodH = np.where(errors <= ecRANSAC)[0]
        if len(possGoodH) > len(possGood):
            possGood = copy.deepcopy(possGoodH)
    return possGood
def ReadCdeTxt(pathFile): # read GCPs 202201301052
    ''' comments:
    .- input pathFile is a string
    .- output codes is a string-list
    .- output xs, ys and zs are float-dictionaries for codes
    .- output switchs is a string-dictionary for codes
    .- output areCodesOn is a boolean-dictionary for codes
    '''
    codes = ReadRectangleFromTxt(pathFile, {'c1':1, 'valueType':'str'}) # important not to sort
    assert len(codes) == len(list(set(codes)))
    rawData = np.asarray(ReadRectangleFromTxt(pathFile, {'c0':1, 'c1':4, 'valueType':'float'}))
    xs, ys, zs = {}, {}, {}
    for pos in range(len(codes)): # important not to sort
        xs[codes[pos]], ys[codes[pos]], zs[codes[pos]] = [rawData[pos, item] for item in range(3)]
    rawData = ReadRectangleFromTxt(pathFile, {'c0':4, 'c1':5, 'valueType':'str'})
    switchs, areCodesOn = {}, {}
    for pos in range(len(codes)): # important not to sort
        assert rawData[pos] in ['on', 'off']
        switchs[codes[pos]], areCodesOn[codes[pos]] = rawData[pos], rawData[pos] == 'on'
    return codes, xs, ys, zs, switchs, areCodesOn
def ReadCdgTxt(pathCdgTxt, options={}): # 202110051016
    ''' comments:
    .- input pathCdgTxt is a string
    .- output cs, rs, xs, ys and zs are float-ndarrays (that can be empty)
    .- output codes is a string-list or None
    '''
    keys, defaultValues = ['readCodes', 'readOnlyGood'], [False, True]
    options = CompleteADictionary(options, keys, defaultValues)
    rawData = np.asarray(ReadRectangleFromTxt(pathCdgTxt, {'c1':5, 'valueType':'float'}))
    if len(rawData) == 0: # exception required
        cs, rs, xs, ys, zs = [np.asarray([]) for item in range(5)]
    else:
        cs, rs, xs, ys, zs = [rawData[:, item] for item in range(5)]
        if options['readOnlyGood']: # disregards negative pixels
            possGood = np.where((cs >= 0.) & (rs >= 0.))[0]
            cs, rs, xs, ys, zs = [item[possGood] for item in [cs, rs, xs, ys, zs]]
    if options['readCodes']:
        codes = ReadRectangleFromTxt(pathCdgTxt, {'c0':5, 'c1':6, 'valueType':'str'}) # can be []
        if len(codes) > 0 and options['readOnlyGood']:
            codes = [codes[pos] for pos in possGood]
    else:
        codes = None
    return cs, rs, xs, ys, zs, codes
def ReadCdhTxt(pathCdhTxt, options={}): # 202110051054
    ''' comments:
    .- input pathCdhTxt is a string
    .- output chs and rhs are float-ndarrays (that can be empty)
    '''
    keys, defaultValues = ['readOnlyGood'], [True]
    options = CompleteADictionary(options, keys, defaultValues)
    rawData = np.asarray(ReadRectangleFromTxt(pathCdhTxt, {'c1':2, 'valueType':'float'}))
    if len(rawData) == 0: # exception required
        chs, rhs = [np.asarray([]) for item in range(2)]
    else:
        chs, rhs = [rawData[:, item] for item in range(2)]
        if options['readOnlyGood']: # disregards negative pixels
            possGood = np.where((chs >= 0.) & (rhs >= 0.))[0]
            chs, rhs = [item[possGood] for item in [chs, rhs]]
    return chs, rhs
def ReadRectangleFromTxt(pathFile, options={}): # 202109141200 # *** 
    assert os.path.isfile(pathFile)
    keys, defaultValues = ['c0', 'c1', 'r0', 'r1', 'valueType', 'nullLine'], [0, 0, 0, 0, 'str', None]
    options = CompleteADictionary(options, keys, defaultValues)
    openedFile = open(pathFile, 'r')
    listOfLines = openedFile.readlines()
    if options['nullLine'] is not None:
        listOfLines = [item for item in listOfLines if item[0] != options['nullLine']]
    if not (options['r0'] == 0 and options['r1'] == 0): # if r0 == r1 == 0 it loads all the rows
        listOfLines = [listOfLines[item] for item in range(options['r0'], options['r1'])]
    for posOfLine in range(len(listOfLines)-1, -1, -1):
        if listOfLines[posOfLine] == '\n':
            print('... line {:5} is empty'.format(posOfLine))
            del listOfLines[posOfLine]
    stringsInLines = [item.split() for item in listOfLines]
    rectangle = stringsInLines
    if not (options['c0'] == options['c1'] == 0): # if c0 == c1 == 0 it loads all the columns
        rectangle = [item[options['c0']:options['c1']] for item in rectangle]
    if options['valueType'] == 'str':
        pass
    elif options['valueType'] == 'float':
        rectangle = [[float(item) for item in line] for line in rectangle]
    elif options['valueType'] == 'int':
        rectangle = [[int(item) for item in line] for line in rectangle]
    else:
        assert False
    if options['c1'] - options['c0'] == 1: # one column
        rectangle = [item[0] for item in rectangle]
    if options['r1'] - options['r0'] == 1: # one row
        rectangle = rectangle[0]
    return rectangle
def WriteCdgTxt(pathCdgTxt, cs, rs, xs, ys, zs, options={}): # 202110051016
    ''' comments:
    .- input pathCdgTxt is a string
    .- input cs, rs, xs, ys and zs are float-ndarrays
    '''
    keys, defaultValues = ['codes'], None
    options = CompleteADictionary(options, keys, defaultValues)
    MakeFolder(pathCdgTxt[0:pathCdgTxt.rfind(os.sep)])
    fileout = open(pathCdgTxt, 'w')
    for pos in range(len(cs)):
        fileout.write('{:15.3f} {:15.3f} {:15.3f} {:15.3f} {:15.3f}'.format(cs[pos], rs[pos], xs[pos], ys[pos], zs[pos]))
        if options['codes'] is None or options['codes'][pos] == 'c,':
            fileout.write(' \t c, r, x, y and z\n')
        else:
            fileout.write(' {:>15} \t c, r, x, y, z and code\n'.format(options['codes'][pos]))
    fileout.close()
    return None
def WriteCdhTxt(pathCdhTxt, chs, rhs): # 202110150951
    ''' comments:
    .- input pathCdhTxt is a string
    .- input chs and rhs are float-ndarrays or float-lists of the same length
    '''
    assert len(chs) == len(rhs) # can be 0
    MakeFolder(pathCdhTxt[0:pathCdhTxt.rfind(os.sep)])
    fileout = open(pathCdhTxt, 'w')
    for pos in range(len(chs)):
        fileout.write('{:15.3f} {:15.3f} \t c and r in the horizon\n'.format(chs[pos], rhs[pos]))
    fileout.close()
    return None
def XYZ2A0(xs, ys, zs): # 202201250808
    '''
    .- input xs, ys and zs are float-ndarrays of the same length
    .- output A0 is a (2*len(xs)x8)-float-ndarray
    '''
    poss0, poss1 = Poss0AndPoss1(len(xs))
    A0 = np.zeros((2 * len(xs), 8))
    A0[poss0, 0], A0[poss0, 1], A0[poss0, 2], A0[poss0, 3] = xs, ys, zs, np.ones(xs.shape)
    A0[poss1, 4], A0[poss1, 5], A0[poss1, 6], A0[poss1, 7] = xs, ys, zs, np.ones(xs.shape)
    return A0
def XYZPa112CURU(xs, ys, zs, Pa11): # *** 
    ''''
    .- input xs, ys and zs are float-ndarrays of the same length
    .- input Pa11 is a 11-float-ndarray
    .- output cUs and rUs are float-ndarrays of the same length as xs
    '''
    dens = Pa11[8] * xs + Pa11[9] * ys + Pa11[10] * zs + 1.
    cUs = (Pa11[0] * xs + Pa11[1] * ys + Pa11[2] * zs + Pa11[3]) / dens
    rUs = (Pa11[4] * xs + Pa11[5] * ys + Pa11[6] * zs + Pa11[7]) / dens
    return cUs, rUs
def Xi2XForParabolicDistortion(xis): # *** 
    ''' comments:
    .- input xis is a float-ndarray
    .- output xs is a float-ndarray
        .- it is solved: xis = xs ** 3 + 2 * xs ** 2 + xs
    '''
    p, qs, Deltas = -1. /3., -(xis + 2. / 27.), (xis + 4. / 27.) * xis
    if np.max(Deltas) < 0: # for xis in (-4/27, 0)
        n3s = (qs + 1j * np.sqrt(np.abs(Deltas))) / 2.
        ns = np.abs(n3s) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3s)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
    elif np.min(Deltas) >= 0: # for xis not in (-4/27, 0)
        auxs = (qs + np.sqrt(Deltas)) / 2.
        ns = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    else:
        possN, possP, ns = np.where(Deltas < 0)[0], np.where(Deltas >= 0)[0], np.zeros(xis.shape) + 1j * np.zeros(xis.shape)
        n3sN = (qs[possN] + 1j * np.sqrt(np.abs(Deltas[possN]))) / 2.
        ns[possN] = np.abs(n3sN) ** (1. / 3.) * np.exp(1j * (np.abs(np.angle(n3sN)) + 2. * np.pi * 1.) / 3.) # we ensure theta > 0; + 2 pi j for j = 0, 1, 2
        auxs = (qs[possP] + np.sqrt(Deltas[possP])) / 2.
        ns[possP] = np.sign(auxs) * (np.abs(auxs) ** (1. / 3.))
    xs = np.real(p / (3. * ns) - ns - 2. / 3.)
    assert np.allclose(xs ** 3 + 2 * xs ** 2 + xs, xis) # avoidable
    return xs
