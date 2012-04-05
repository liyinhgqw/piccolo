#ifndef _IMAGENET_H_

#define _IMAGENET_H_

#define TARGET_HIGH 0.9
#define TARGET_LOW 0.1
#include "backprop.hpp"
extern "C" {
	#include "pgmimage.h"
}

class ImageNet {

	public:
		void load_target(IMAGE*,BPNN*);
		void load_input_with_image(IMAGE*,BPNN*);

};
#endif	/* _IMAGENET_H_ */
