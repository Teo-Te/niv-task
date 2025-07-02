# Audio EnCodec

## What was this task about, according to as much as I have understood

It has to do with compressing an audio file for transfering it and then when
decoding it all this using an intelligent AI model, then it will turn back to the same audio

## What is EnCodec

It's an audio codec developed by facebook which uses neural networks, it can
compress an audio file and then reconstruct it again.

### Step 1: Converting

- **FFmpeg** converts the audio file to the specific 24 kHz .wav format because as per the EnCodec
  doc on github "A causal model operating at 24 kHz". I believe that is what needed to be used.

### Step 2: EnCodec Encoding

- First cloned the EnCodec repo and then imported it (mind that it has my local root folder in there)
  After the model from EnCodec has succeffully then then actual encoding happens

### Step 4: The code is sent to the second server

- Make an http request and send the code to the other backend to decode it.

### Step 5: EnCodec Decoding

- The compressed codes are received on the second server there they are reconstructed
  and then the codes are turned back into audio: (24kHz to 22050Hz (as per the requirements))

### Upload audio to frontend (YES)

### Convert audio to 24kHz waveform using ffmpeg (YES)

### Encode with EnCodec (use ONNX Runtime javascript) (YES)

### Send it to another server (YES)

### On the other server: Create a post route to receive input (YES)

### Read the input (8xN matrix encodec) (YES)

### Decode it with encodec (YES)

### Save as 22050Hz wav (YES)

## Special Notes

I did do this task cause that is what was given to me to prove myself a little.
I know I do not know much about audios and how does the AI world in itself work,
and yes ofc, **THIS** should be what I was supposed to learn at the internship
not doing WordPress to be honest. Thanks anyways.
