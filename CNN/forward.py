import numpy as np

def convolution(image, filt, bias, s=1):
  
    (no_of_filters, n_c_f, filter_dim, _) = filt.shape # filter dimensions
    #(8, 1, 5, 5) (8, 8, 5, 5)
    #n_c_f = ? 

    n_c, image_dim, _ = image.shape # image dimensions 
    #(1, 28, 28) (8, 24, 24)
    #n_c = ?
    
    out_dim = int((image_dim - filter_dim)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((no_of_filters,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    #curr_f = for indexing filter and bias
    #curr_y, curr_x = for indexing image on the basis of stride
    #out_y, out_x = for indexing the output
    for curr_f in range(no_of_filters):

        curr_y = out_y = 0

        while curr_y + filter_dim <= image_dim:

            curr_x = out_x = 0

            while curr_x + filter_dim <= image_dim:

                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+filter_dim, curr_x:curr_x+filter_dim]) + bias[curr_f]
                curr_x += s
                out_x += 1
                
            curr_y += s
            out_y += 1
    
    return out #(8, 24, 24) (8, 20, 20)

def maxpool(image, kernal_size=2, s=2):
    n_c, input_columns, input_rows = image.shape
    #(8, 24, 24) (8, 20, 20)
    
    output_columns = int((input_columns - kernal_size)/s)+1
    output_rows = int((input_rows - kernal_size)/s)+1
    
    downsampled = np.zeros((n_c, output_columns, output_rows))

    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        #curr_y, curr_x = for indexing image
        #out_x, out_y = for indexing the output
        curr_y = out_y = 0
        while curr_y + kernal_size <= input_columns:
            curr_x = out_x = 0
            while curr_x + kernal_size <= input_rows:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+kernal_size, curr_x:curr_x+kernal_size])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return downsampled #(8, 10, 10)

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    # print(probs.shape)
    # print(label.shape)
    return -np.sum(label * np.log(probs))