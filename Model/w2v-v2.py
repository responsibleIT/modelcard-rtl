import time

import argparse
import sys
import re
from typing import Dict

import torch
from datasets import Audio, Dataset, load_dataset, load_metric

from pyctcdecode import BeamSearchDecoderCTC
from transformers import AutoFeatureExtractor, AutoTokenizer, pipeline
import transformers
import librosa

# model id
model_id = '../models/xls-r-2b-nl-v2_lm-5gram-os'

# load processor
sampling_rate = 16000 # I recommend this, much faster than 44.1kHz without sacrificing much quality

# resample audio
wav_file = "fn000029.wav" # Or some other .wav audio file
sound_array, _ = librosa.load(wav_file, sr = sampling_rate) # Loading audio with librosa to a 'sound array', so just a long list of 16000 numbers per second 

# load eval pipeline
device = 1 if torch.cuda.is_available() else -1 # Running on GPU (0) if pytorch detects that you have it, otherwise CPU (-1)

print('GPU id={} (-1 if there is no GPU)'.format(device))

# st = time.time()

# # individually initializing all the components of speech recognition
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
# config = transformers.PretrainedConfig.from_pretrained(model_id)
# model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# processor = transformers.AutoProcessor.from_pretrained(model_id)
# language_model = BeamSearchDecoderCTC.model_container[processor.decoder._model_key]._kenlm_model

# # initializing the pipeline 
# asr = pipeline("automatic-speech-recognition", config=config, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=processor.decoder, device=device, return_timestamps="word")

# print('model loaded in {} seconds'.format(str(time.time() - st)))

# # map function to decode audio
# def map_to_pred(sound_array, chunk_length_s = 10, stride_length_s = 2):
#     return asr(sound_array, chunk_length_s = chunk_length_s, stride_length_s = stride_length_s)

# st = time.time()

# prediction = map_to_pred(sound_array)

# print('first pred parsed in {} seconds'.format(str(time.time() - st)))
# print('now going for the additional information:')

# st = time.time()

import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import kenlm as kl
import operator
import functools
import json
import logging
from transformers import (AutoFeatureExtractor, 
                            PretrainedConfig, 
                            Wav2Vec2ForCTC, 
                            AutoTokenizer, 
                            AutoProcessor,
                            pipeline,
                            AutomaticSpeechRecognitionPipeline,
                            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
                            MODEL_FOR_CTC_MAPPING
                    )

from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modelcard import ModelCard
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.pipelines.base import ArgumentHandler, infer_framework_load_model
import numpy as np
import pandas as pd

        
class ASRUncertaintyPipeline(AutomaticSpeechRecognitionPipeline):
            """
            Pipeline that aims at extracting spoken text contained within some audio.

            The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
            to support multiple audio formats

            This pipeline can also do uncertainty analysis, for that the __init__ and postprocess functions are overwritten.

            Arguments:
                model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
                    The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
                    [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
                tokenizer ([`PreTrainedTokenizer`]):
                    The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
                    [`PreTrainedTokenizer`].
                feature_extractor ([`SequenceFeatureExtractor`]):
                    The feature extractor that will be used by the pipeline to encode waveform for the model.
                chunk_length_s (`float`, *optional*, defaults to 0):
                    The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default). Only
                    available for CTC models, e.g. [`Wav2Vec2ForCTC`].
                    <Tip>
                    For more information on how to effectively use `chunk_length_s`, please have a look at the [ASR chunking
                    blog post](https://huggingface.co/blog/asr-chunking).
                    </Tip>
                stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
                    The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables
                    the model to *see* more context and infer letters better than without this context but the pipeline
                    discards the stride bits at the end to make the final reconstitution as perfect as possible.
                    <Tip>
                    For more information on how to effectively use `stride_length_s`, please have a look at the [ASR chunking
                    blog post](https://huggingface.co/blog/asr-chunking).
                    </Tip>
                framework (`str`, *optional*):
                    The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
                    installed. If no framework is specified, will default to the one currently installed. If no framework is
                    specified and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if
                    no model is provided.
                device (`int`, *optional*, defaults to -1):
                    Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on
                    the associated CUDA device id.
                decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*):
                    [PyCTCDecode's
                    BeamSearchDecoderCTC](https://github.com/kensho-technologies/pyctcdecode/blob/2fd33dc37c4111417e08d89ccd23d28e9b308d19/pyctcdecode/decoder.py#L180)
                    can be passed for language model boosted decoding. See [`Wav2Vec2ProcessorWithLM`] for more information.
                uncertainty_analysis (`bool`, *optional*, defaults to `False`):
                    Boolean that holds whether uncertainty analysis results will be computed and given back. Using this will make 
                    prediction significantly slower.
                return_timestamps (`str`, *optional*, defaults to `word`):
                    Whether or not to return the timestamps associated with the individual words.

            """
            def __init__(self, 
                model: Union["PreTrainedModel", "TFPreTrainedModel"],
                tokenizer: Optional[PreTrainedTokenizer] = None,
                feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
                modelcard: Optional[ModelCard] = None,
                framework: Optional[str] = None,
                lm: Optional[kl.LanguageModel] = None,
                task: str = "",
                args_parser: ArgumentHandler = None,
                device: int = -1,
                binary_output: bool = False,
                uncertainty_analysis: bool = False,
                beam_prune_logp: float = -10.0,
                token_min_logp: float = -5.0,
                return_timestamps: str = "word",
                         **kwargs,
            ):
                """ This function is adapted from the __init__ in class Pipeline, the __init__ in class AutomaticSpeechRecognitionPipeline and has 
                two extra vars for uncertainty estimation. """
                import operator
                import functools
                if framework is None:
                    framework, model = infer_framework_load_model(model, config=model.config)

                self.task = task
                self.model = model
                self.tokenizer = tokenizer
                self.feature_extractor = feature_extractor
                self.modelcard = modelcard
                self.framework = framework
                self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
                    
                self.binary_output = binary_output

                # Special handling
                self.model = self.model.to(self.device)

                # Update config with task specific parameters
                task_specific_params = self.model.config.task_specific_params
                if task_specific_params is not None and task in task_specific_params:
                    self.model.config.update(task_specific_params.get(task))

                self.call_count = 0
                self._batch_size = kwargs.pop("batch_size", None)
                self._num_workers = kwargs.pop("num_workers", None)
                self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
                self.check_model_type(dict(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items() + MODEL_FOR_CTC_MAPPING.items()))
                
                # Two extra hyperparameters
                self.beam_prune_logp = beam_prune_logp 
                self.token_min_logp = token_min_logp
                
                # THESE TWO LINES ARE ADDED FROM THE ORIGINAL CODE
                # original __init__ function is from transformers 4.21.1 and is a concatenation of the __init__ of class Pipeline and class AutomaticSpeechRecognitionPipeline
                self.uncertainty_analysis = uncertainty_analysis
                self.return_timestamps = return_timestamps

                if (
                    self.feature_extractor._processor_class
                    and self.feature_extractor._processor_class.endswith("WithLM")
                    and kwargs.get("decoder", None) is not None
                ):
                    self.decoder = kwargs["decoder"]
                    self.type = "ctc_with_lm"
                    self.lm = lm

                else:

                    raise ValueError("You are using a custom pipeline for uncertainty estimation. It can only be used with a CTCWithLM-type of model.")



            def postprocess(self, model_outputs):
                """ Adapted from AutomaticSpeechRecognitionPipeline postprocess() in transformers 4.21.1 to make room for the uncertainty analysis variables."""
                
                # first some functions to be able to calculate the metrics, these 5 functions are copied from: videopipe/pipelines/midroll_marker/self_similarity_novelty_peaks.py 
                # adapted the get_ssm_audio_tags() to get_ssm_audio_tags_fast() for faster computations (only computes needed entries in the matrix with some buffer)
                def unit_vector(vector):
                    """ Returns the unit vector of the vector.  """
                    return vector / np.linalg.norm(vector)


                def angle_between(v1, v2):
                    """ Returns the angle in radians between vectors 'v1' and 'v2'::

                            >>> angle_between((1, 0, 0), (0, 1, 0))
                            1.5707963267948966
                            >>> angle_between((1, 0, 0), (1, 0, 0))
                            0.0
                            >>> angle_between((1, 0, 0), (-1, 0, 0))
                            3.141592653589793
                    """
                    v1_u = unit_vector(v1)
                    v2_u = unit_vector(v2)
                    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


                def compute_novelty_ssm(S, kernel=None, L=4, var=0.5, exclude=False):
                    """Compute novelty function from SSM [FMP, Section 4.4.1]

                    Notebook: C4/C4S4_NoveltySegmentation.ipynb
                    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html

                    Args:
                        S: SSM
                        kernel: Checkerboard kernel (if kernel==None, it will be computed)
                        L: Parameter specifying the kernel size M=2*L+1
                        var: Variance parameter determing the tapering (epsilon)
                        exclude: Sets the first L and last L values of novelty function to zero

                    Returns:
                        nov: Novelty function
                    """
                    if kernel is None:
                        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
                    N = S.shape[0]
                    M = 2 * L + 1
                    nov = np.zeros(N)
                    # np.pad does not work with numba/jit
                    S_padded = np.pad(S, L, mode='constant')

                    for n in range(N):
                        # Does not work with numba/jit
                        nov[n] = np.sum(S_padded[n:n + M, n:n + M] * kernel)
                    if exclude:
                        right = np.min([L, N])
                        left = np.max([0, N - L])
                        nov[0:right] = 0
                        nov[left:N] = 0

                    return nov

                def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
                    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
                    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

                    Notebook: C4/C4S4_NoveltySegmentation.ipynb
                    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html

                    Args:
                        L: Parameter specifying the kernel size M=2*L+1
                        var: Variance parameter determing the tapering (epsilon)

                    Returns:
                        kernel: Kernel matrix of size M x M
                    """
                    taper = np.sqrt(1 / 2) / (L * var)
                    axis = np.arange(-L, L + 1)
                    gaussian1D = np.exp(-taper**2 * (axis**2))
                    gaussian2D = np.outer(gaussian1D, gaussian1D)
                    kernel_box = np.outer(np.sign(axis), np.sign(axis))
                    kernel = kernel_box * gaussian2D
                    if normalize:
                        kernel = kernel / np.sum(np.abs(kernel))
                    return kernel

                def get_ssm_audio_tags_fast(audio_tagging_df, kernelsize=10):
                    """ Calculate the self similarity matrix given the audio_tagging_df data model data as df
                    This function is adapted in such a way that it will only calculate the selfsimilarity between columns closer than
                    `kernelsize` to it. 
                    Args:
                        audio_tagging_df: the data field of the audio_tagging_df datamodel as a dataframe
                        kernelsize: how far away can the other samples be, to make sure we don't calculate values we won't need.
                    Returns:
                        ssm: a numpy 2d array that is the self similarity matrix
                        frames_ix: corresponding list of dimension_idx
                    """
                    size = audio_tagging_df.shape[0]
                    ssm = np.array([[angle_between(audio_tagging_df[j], audio_tagging_df[i]) if (j< i+kernelsize) and (j > i-kernelsize) else 0 for j in range(size)] for i in range(size)])
                    return ssm #, audio_tagging_df.index

                  
                # piece of code copied from original postprocess() function. Unpacks the logits and feeds them through the decoder
                final_logits = []
                for outputs in model_outputs:
                    logits = outputs["logits"].numpy()
                    stride = outputs.get("stride", None)
                    if stride is not None:
                        total_n, left, right = stride
                        # Total_n might be < logits.shape[1]
                        # because of padding, that's why
                        # we need to reconstruct this information
                        # This won't work with left padding (which doesn't exist right now)
                        right_n = total_n - right
                        logits = logits[:, left:right_n]
                    final_logits.append(logits)
                logits = np.concatenate(final_logits, axis=1)
                logits = logits.squeeze(0)
                
                
                # BEAM SEARCH 
                beams = self.decoder.decode_beams(logits, token_min_logp = self.token_min_logp, beam_prune_logp = self.beam_prune_logp)
                 
                # unpack the decoder results
                text = beams[0][0]
                text_frames = beams[0][2]

                # if we don't do uncertainty analysis & don't give timestamps back, this is the output
                output = text_frames
                
                if self.uncertainty_analysis:
                    # calculate the score the lm gives to each individual word. I.e. how likely is this combination of words?
                    lm_wordscores = list(self.lm.full_scores(beams[0][0]))
                if self.return_timestamps:
                    # calculate the timestamps
                    # Simply cast from pyctcdecode format to wav2vec2 format to leverage
                    # pre-existing code later
                    chunk_offset = beams[0][2]
                    word_offsets = []
                    for word, (start_offset, end_offset) in chunk_offset:
                        word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                    
                    offsets = word_offsets
                    
                    chunks = []
                    
                    inputs_to_logits_ratio = functools.reduce(operator.mul, self.model.config.conv_stride, 1)
                    for item in offsets:
                        start = item["start_offset"] * inputs_to_logits_ratio
                        start /= self.feature_extractor.sampling_rate

                        stop = item["end_offset"] * inputs_to_logits_ratio
                        stop /= self.feature_extractor.sampling_rate

                        chunks.append({"word": item[self.return_timestamps], "start_time": start, "end_time": stop})
                    output = chunks
                    
                if self.uncertainty_analysis:      
                    # calculate the logit-based uncertainty analysis parameters per word
                    analyzed = []
                    length_segment = logits.shape[0]
                    end_previous = 0
                    
                    # loop through all the words
                    for i in range(len(text_frames)):
                        # find the location of this word in the logits
                        frame_start = text_frames[i][1][0]
                        frame_end = text_frames[i][1][1]
                        word = text_frames[i][0]
                        
                        # get the logits of this word and some buffering around it
                        indexing_correction = max(end_previous,frame_start-20)
                        logits_word = logits[indexing_correction:min(frame_end + 20,length_segment),:]
                        
                        # find the exact indices of the word within the buffered logit array
                        start_word = frame_start - indexing_correction
                        end_word = frame_end - indexing_correction
                        end_previous_corrected = end_previous - indexing_correction

                        # compute novelty function (that says something about the similarity to frames around it)
                        ssm = get_ssm_audio_tags_fast(logits_word.transpose())   
                        novel = compute_novelty_ssm(ssm)
                        
                        # calculate the three logit-based uncertainty measures
                        # silence within: do a rolling min on the novelty function, if the max of that is above/below (?) a certain value it is likely that there is a silence within
                
                        silence_within = np.max(pd.Series(novel[start_word:end_word]).rolling(7).min())
                  
                        # missed_speech_before: the mean value of the novelty function between words, if this is above/below (?) a certain value it is likely that there is a word missed in that period
                        missed_speech_before = np.mean(pd.Series(novel[end_previous_corrected:start_word]))
                      
                        # logits_median: get the median value of all the logit values in the word, if these are below a certain value it is likely that the speech was unintelligible
                        logits_median = np.quantile(logits_word[start_word:end_word,:],0.5)
                        word_record = {"word":word,
                                        "lm_score": lm_wordscores[i][0],
                                        "ngram_size": lm_wordscores[i][1],
                                        "oov": lm_wordscores[i][2],
                                        "silence_within": silence_within, 
                                        "missed_speech_before": missed_speech_before, 
                                        "logits_median":logits_median}
                        if self.return_timestamps:
                            word_record = {**word_record, **{ 
                                        "start_time":chunks[i]["start_time"],
                                        "end_time":chunks[i]["end_time"]}}
                        else:
                            word_record = {**word_record, **{ 
                                        "start_frame":frame_start,
                                        "end_frame":frame_end}}
                          
                        analyzed += [word_record]

                        # prepare for next round
                        end_previous = frame_end
                    output = analyzed
                return output #analyzed


# Detect and set device paramaters
device = "cpu"
if torch.cuda.is_available():
    # Check if there is enough memory to load in on the GPU, if not, load it on the cpu
    total_GPU_mem_mb = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
    reserved_GPU_mem_mb = torch.cuda.memory_reserved(0) // 1024 ** 2
    print(f"Before loading model there is {reserved_GPU_mem_mb/1024.0:.2f}Gb/{total_GPU_mem_mb/1024.0:.2f}Gb available on GPU.")
    if total_GPU_mem_mb >= 10000:
        print("Enough memory available to load speech recognition on GPU.")
        device = "cuda"
    else:
        print("There is a GPU, but not enough memory available to load it on there.")

device_id = 0 if device == "cuda" else -1
logging.info(f"Using detected device (device_id = {device_id}) for inference")    

# Get individual components
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
config = PretrainedConfig.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
lm = BeamSearchDecoderCTC.model_container[processor.decoder._model_key]._kenlm_model

# Location of the lm.bin
# lm = "/dbfs/FileStore/Iskaj/Models/nl_model_new_params/language_model/lm.bin"

# Load custom made ASR pipe
asr = ASRUncertaintyPipeline(feature_extractor = feature_extractor,
                            config = config, 
                            model = model,
                            tokenizer = tokenizer,
                            decoder = processor.decoder,
                            device = device_id,
                            beam_prune_logp = -15.0, # -7.5
                            token_min_logp = -10,   # -7.5                 
                            return_timestamps= "word",
                            uncertainty_analysis = True,
                            lm = lm)

prediction = asr(sound_array, chunk_length_s = 10, stride_length_s = 2)
prediction
st = time.time()
print('final prediction parsed in {} seconds'.format(str(time.time() - st)))

import json

# store prediction
with open('prediction.json', 'w') as fp:
    json.dump(prediction, fp)

#     loading
# with open('data.json', 'r') as fp:
#     data = json.load(fp)