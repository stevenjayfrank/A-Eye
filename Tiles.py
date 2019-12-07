def getEntropy(image):  #image is a numpy array
    imdata = image.flatten()
    data_length = imdata.size
    data_set = list(set(imdata))
    probs = [np.size(imdata[imdata == i])/(1.0 * data_length) for i in data_set]
    ent = np.sum([p * np.log2(1.0 / p) for p in probs])
    return ent


def makeTiles(filename, shortname, tilesize, path, overlap):
    im_jpg = Image.open(filename)
    im_arr = imageio.imread(filename)
    image_dims = im_arr.shape[:2]
    keeps = []
    min_ent = getEntropy(im_arr)
    for i in range(0, image_dims[0], int(tilesize/overlap)):
        for j in range(0, image_dims[1], int(tilesize/overlap)):
            tile = im_jpg.crop((i,j,i+tilesize,j+tilesize))                    
            if getEntropy(np.array(tile)) > min_ent:
                keeps.append(tile)
    for k in range(len(keeps)):
        keeps[k].save(path+shortname+"_tile_"+str(k)+".jpg")

