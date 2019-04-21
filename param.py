letter_string = '0123456789' \
                + 'abcdefghijklmnopqrstuvwxyz' \
                + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '

characters = [letter for letter in letter_string]
classes = len(characters) + 1

alphabet = {character: index for character, index in zip(characters,
            range(classes))}
rev_alphabet = {index: character for character, index in alphabet.items()}

img_w = 100
img_h = 32

img_input_shape = (img_w, img_h, 1)
max_string_len = 16

# Training specific parameters
time_steps = 24
epochs = 5
batch_size = 256
validation_batch_size = 64
