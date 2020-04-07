
import sys
import csv
import subprocess
import math
import json
import os
import shlex
from optparse import OptionParser
from pydub.silence import detect_nonsilent
from pydub.silence import detect_silence
from pydub import AudioSegment
import moviepy.editor
import itertools
import heapq
from google.cloud import storage
import re
# import nltk
from tinytag import TinyTag
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def getAudioFromVideo(videoTitle):
    # video = moviepy.editor.VideoFileClip(videoTitle)
    # audio = video.audio
    # # Replace the parameter with the location along with filename
    # audio.write_audiofile("sample.wav")
    # ffmpeg -i video.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav
    import subprocess
    cmd = "ffmpeg -i "+videoTitle+" -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    cmd1 = "ffmpeg -i "+videoTitle+" -ab 160k -ac 2 -ar 44100 -vn audio.mp3"
    command = cmd
    os.system(cmd)
    os.system(cmd1)

#Split by manifest - segments based on info in manifest.csv
def split_by_manifest(filename, manifest,
                      extra="", **kwargs):
    """ Split video into segments based on the given manifest file.
    Arguments:
        filename (str)      - Location of the video.
        manifest (str)      - Location of the manifest file.
        vcodec (str)        - Controls the video codec for the ffmpeg video
                            output.
        acodec (str)        - Controls the audio codec for the ffmpeg video
                            output.
        extra (str)         - Extra options for ffmpeg.
    """
    if not os.path.exists(manifest):
        print("File does not exist: %s" % manifest)
        raise SystemExit

    with open(manifest) as manifest_file:
        manifest_type = manifest.split(".")[-1]
        if manifest_type == "json":
            config = json.load(manifest_file)
        elif manifest_type == "csv":
            config = csv.DictReader(manifest_file)
        else:
            print("Format not supported. File must be a csv or json file")
            raise SystemExit

        split_cmd = ["ffmpeg", "-i", filename] + shlex.split(extra)
        try:
            fileext = filename.split(".")[-1]
        except IndexError as e:
            raise IndexError("No . in filename. Error: " + str(e))
        for video_config in config:
            split_str = ""
            split_args = []
            try:
                split_start = video_config["start_time"]
                split_length = video_config.get("end_time", None)
                if not split_length:
                    split_length = video_config["length"]
                filebase = video_config["rename_to"]
                if fileext in filebase:
                    filebase = ".".join(filebase.split(".")[:-1])

                split_args += ["-ss", str(split_start), "-t",
                    str(split_length), filebase + "." + fileext]
                print("########################################################")
                print("About to run: "+" ".join(split_cmd+split_args))
                print("########################################################")
                subprocess.check_output(split_cmd+split_args)
            except KeyError as e:
                print("############# Incorrect format ##############")
                if manifest_type == "json":
                    print("The format of each json array should be:")
                    print("{start_time: <int>, length: <int>, rename_to: <string>}")
                elif manifest_type == "csv":
                    print("start_time,length,rename_to should be the first line ")
                    print("in the csv file.")
                print("#############################################")
                print(e)
                raise SystemExit


#reading mp3
def SegmentVideo(VideoTitle,Title):
	audio_segment = AudioSegment.from_mp3(Title)
	print("--generating silent ranges--")    
	silent_ranges = detect_silence(audio_segment, min_silence_len=10000)

	len_seg = len(audio_segment)

	# if there is no silence, the whole thing is nonsilent
	if not silent_ranges:
	  nonsilent_ranges=[[0, len_seg]]

	# short circuit when the whole audio segment is silent
	if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
	  nonsilent_ranges=[]

	prev_end_i = 0
	nonsilent_ranges = []

	for start_i, end_i in silent_ranges:
	  nonsilent_ranges.append([prev_end_i, start_i])
	  prev_end_i = end_i

	if end_i != len_seg:
	  nonsilent_ranges.append([prev_end_i, len_seg])
	  
	if nonsilent_ranges[0] == [0, 0]:
	  nonsilent_ranges.pop(0)

	segs = silent_ranges + nonsilent_ranges
	print("-----------------------------------------")
	print("Silent ranges",silent_ranges)
	print("Non Silent ranges",nonsilent_ranges)
	print("All segments",segs)
	print("Number of segments",len(segs))


	with open('manifest.csv', 'w', newline='') as file:
	  writer = csv.writer(file)
	  writer.writerow(["start_time", "length", "rename_to"])
	  k=0
	  for i in segs:
	    a=int(i[0]/1000)
	    b=int(i[1]/1000)
	    dur=b-a
	    k=k+1
	    writer.writerow([a,dur,"segments"+str(k)])

	#generating segments:
	split_by_manifest(VideoTitle,"manifest.csv")

def transcript_extract(storage_uri,j,segemnt_transcripts,segment_durations):
    client = speech_v1.SpeechClient()
    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'
    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100
    # The language of the supplied audio
    language_code = "en-IN"
    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "encoding": encoding,
    "audio_channel_count":2
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    text=""
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        a=u" {}".format(alternative.transcript)
        text=text+a+"."
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    word_count = 0
    for word in nltk.word_tokenize(text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                    word_count+=1
                else:
                    word_frequencies[word] += 1
                    word_count+=1    
    segemnt_transcripts.append([j,segment_durations[j-1][0],text,segment_durations[j-1][1],len(nltk.sent_tokenize(text)),word_count])


def RankSegments():
    path = os.getcwd()
    files = []
    for r, d, f in os.walk(path):
        for file in f:
           if re.match(r'segments[0-9]*.mp4',file):
                files.append(file)    
    i=1
    for k in files:
        cmd = "ffmpeg -i "+k+" -ab 160k -ac 2 -ar 44100 -vn "+str(i)+".wav"
        os.system(cmd)
    i=i+1          
    # from google.cloud import storage
    path = os.getcwd()
    wavfiles = []
    for r, d, f in os.walk(path):
        for file in f:
            if re.match(r'[0-9]*.wav',file):
                wavfiles.append(file)
    print(len(wavfiles))            
    # os.system('export GOOGLE_APPLICATION_CREDENTIALS="tts.json"')            
    for fp in wavfiles:
        storage_client = storage.Client()
        bucket = storage_client.bucket('final_sem')
        blob = bucket.blob(fp)
        blob.upload_from_filename(fp)
        print(
            "File {} uploaded to {}.".format(
                fp, fp
            )   
        )  
    segment_durations = []
    for i in files:
        tag = TinyTag.get(i)
        segment_durations.append([i,tag.duration])   
    segemnt_transcripts = []
    j=1    
    for k in range(1,len(wavfiles)+1):
            transcript_extract('gs://final_sem/'+str(k)+'.wav',j,segemnt_transcripts,segment_durations)
            j=j+1        
    features = segemnt_transcripts
    # features = sorted(features, key = lambda x : (x[2],x[3],x[4]), reverse=True)
    print(features)
    score = 0
    for i in features:
      score+=i[3]+i[4]+i[5]
      i.append(i[3]+i[4]+i[5])
    features = sorted(features, key = lambda x : x[6], reverse=True)
    for i in features:
      s = i[1]
      s = s.split('.')
      s = s[0].split('s')
      i[0] = int(s[2]) 
    final_feature = []
    mean = score / len(features)
    for i in features:
      if(i[6]>=mean):
        final_feature.append(i)
    final_feature = sorted(final_feature, key = lambda x : x[0])
    final_transcript = ''
    for i in final_feature:
      final_transcript +=i[2]     
    print(final_transcript)
    with open('transcript.txt', 'w') as f:
      print(final_transcript, file=f)        

def TextSummaryExtractor(TranscriptFile):
    print('Working on generating summary')
    fp = open(TranscriptFile, 'r')
    article_text = fp.read()
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)    
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    nltk.download('punkt')
    nltk.download('stopwords')
    sentence_list = nltk.sent_tokenize(article_text)
    new_sentence_list = []
    h=1
    for i in sentence_list:
        new_sentence_list.append([h,str(h)+":"+i])
        h+=1
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    word_count = 0
    for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                    word_count+=1
                else:
                    word_frequencies[word] += 1
                    word_count+=1
    # print(word_frequencies)
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    sentence_scores = {}
    for sent in new_sentence_list:
    # print(sent)
        for word in nltk.word_tokenize(sent[1].lower()):
          if word in word_frequencies.keys():
                if sent[1] not in sentence_scores.keys():
                    sentence_scores[sent[1]] = word_frequencies[word]
                else:
                    sentence_scores[sent[1]] += word_frequencies[word]
    scores = []
    for sent in sentence_scores:     
      scores.append(sentence_scores[sent])
    mean = sum(scores)/len(scores)
    k = [x for x in scores if (x > (mean-(mean/2)))]
    w = len(k)
    summary_sentences = heapq.nlargest(w, sentence_scores, key=sentence_scores.get)
    new_summary_sentence = []
    for i in summary_sentences:
        k = i.split(':')
        new_summary_sentence.append([int(k[0]),k[1]])  
    new_summary_sentence = sorted(new_summary_sentence, key=lambda x : x[0])
    summary = ''
    for i in new_summary_sentence:
        summary = summary + i[1]
    with open('output_summary.txt', 'w') as f:
        print(summary, file=f)

from google.cloud import speech_v1
from google.cloud.speech_v1 import enums


def extractTranscript(storage_uri):
    """
    Transcribe long audio file from Cloud Storage using asynchronous speech
    recognition

    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
    """

    client = speech_v1.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100

    # The language of the supplied audio
    language_code = "en-IN"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "encoding": encoding,
    "audio_channel_count":2
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    text=""
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        a=u" {}".format(alternative.transcript)
        text=text+a+"."
    f = open("transcript.txt", "w")
    f.write(text)
    f.close()
    # print(text)



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )




def main(videoTitle, option):
    getAudioFromVideo(videoTitle)
    # upload_blob('final_sem','audio.wav','audio3.wav')
    # extractTranscript('gs://final_sem/audio3.wav')
    # extractTranscript(videotitle)
    SegmentVideo(videoTitle,'audio.mp3')
    RankSegments()
    # condenseSegments()
    TextSummaryExtractor('transcript.txt')
    # if(option == 1):
    #     extractImportantChalkBoardRepresentation()
    # elif(option == 2):
    #     extractImportantPPTFrames()        
    # generatePDF()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2]);    