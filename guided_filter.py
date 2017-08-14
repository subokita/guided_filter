#!/usr/bin/env python
# encoding: utf-8

# Based on http://kaiminghe.com/publications/eccv10guidedfilter.pdf
# and https://github.com/pfchai/GuidedFilter

from scipy.misc import imread
import numpy             as np
import scipy.misc        as sci
import matplotlib.pyplot as plot


class GuidedFilter( object ):
    @classmethod
    def _filter( cls, input: np.ndarray, r: int ) -> np.ndarray:
        limit, _    = input.shape
        cumulative  = np.cumsum(input, 0)
        result      = np.zeros_like( input )

        result[0:r+1, :]         = cumulative[r:2*r+1, :]
        result[r+1:limit-r, :]   = cumulative[2*r+1:limit, :] - cumulative[0:limit-2*r-1, :]
        result[limit-r:limit, :] = np.tile( cumulative[limit-1, :], [r, 1] ) - cumulative[limit-2*r-1:limit-r-1, :]

        return result


    @classmethod
    def box_filter( cls, image: np.ndarray, radius: int ) -> np.ndarray:
        # Apply on the row
        output = cls._filter( image, radius ) 

        # ... then transpose and apply on the row (column actually)
        output = cls._filter( output.transpose(), radius ) 
        return output.transpose()


    @classmethod
    def apply(cls, guide: np.ndarray, image: np.ndarray, radius: int, epsilon: float) -> np.ndarray:

        N = cls.box_filter( np.ones_like( image ), radius )

        mean_guide             = cls.box_filter( guide, radius ) / N
        mean_image             = cls.box_filter( image, radius ) / N
        mean_guide_image       = cls.box_filter( guide * image, radius ) / N
        covariance_guide_image = mean_guide_image - mean_guide * mean_image

        mean_squared_guide = cls.box_filter( guide * guide, radius ) / N
        variance_guide     = mean_squared_guide - mean_guide * mean_guide

        a = covariance_guide_image / (variance_guide + epsilon)
        b = mean_image - a * mean_guide

        mean_a = cls.box_filter( a, radius ) / N
        mean_b = cls.box_filter( b, radius ) / N

        return mean_a * guide + mean_b



def main():
    image   = imread( 'cat.bmp' ).astype( np.float32 ) / 255.0
    radius  = 5
    epsilon = 0.05
    result  = GuidedFilter.apply(image, image, radius, epsilon)
    
    plot.imshow( np.concatenate( (image, result ), axis = 1 ), cmap = 'gray' )
    plot.show()

    print('[DONE]')
    return

if __name__ == '__main__':
    main()
