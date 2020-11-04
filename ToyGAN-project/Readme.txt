This Toy Gan project started as an attempt to create a GAN that generates inverted images.
In that effort, I attempted to first create a generator that generates inverted 1 pixel images. (Images consisting of 1 pixel)
And by doing that, I ended up creating a generator that takes a number between/including 0 and 1 and returns 1 minus the number.

# The GAN's Program Description:
# The GAN's Input is a number between 0 and 1.  
# The GAN's generator outputs both the input number and 1 minus the input number.  
# The GAN's discriminator determines whether the two numbers sum to 1.

The key feature of a Generative Adversarial Neural Network (GAN) is that the first part (the generator) of the GAN is trained by training the entire GAN while the second part (the discriminator) of the GAN has its weights frozen. By freezing the second half's (the discriminator's) weights while the entire GAN trains, the first half (the generator) is forced to create only outputs that when input into the second half (discriminator) return a 1 instead of a 0.

