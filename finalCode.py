import sys
import csv
import subprocess
import math
import json
import os
import shlex
from optparse import OptionParser
import cv2
import itertools
import heapq
from google.cloud import storage
import re
import nltk
import fpdf
nltk.download('stopwords')
nltk.download('punkt')
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips, VideoClip
from chalkboard import framegen,chalk,refine_chalks
from pptboard import ppt,refine_ppt
from fpdf import FPDF

cwd = os.getcwd()
print(cwd)
newdir = cwd +"/datadir"
os.chdir(newdir)

def getAudioFromVideo(videoTitle):
    print("-------------Getting Audio From Video-------------")
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
def SegmentVideo(VideoTitle):
    os.system('ffmpeg -i '+VideoTitle+' -af silencedetect=noise=-30dB:d=0.5 -f null - 2> silence_detection.txt')
    cap = cv2.VideoCapture(VideoTitle)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    filepath = 'silence_detection.txt'
    txt=[]
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #print("Line {}: {}".format(cnt, line.strip()))
            txt.append(line)
            line = fp.readline()
            cnt += 1
    l =[]
    print(len(txt))
    for i in txt:
        if "[silencedetect" in i:
            i=i.rstrip()
            i=i.lstrip()
            l.append(i)
    print(len(l))
    m=[]
    for i in l:
        x=i.split(' ')
        #print(x[0],x[1],x[2])
        if(i.startswith(x[0])):
            i=i.replace(x[0],'')
            i=i.strip()
        if(i.startswith(x[1])):
            i=i.replace(x[1],'')
            i=i.strip()
        if(i.startswith(x[2])):
            i=i.replace(x[2],'')
            i=i.strip()
        i=i.rstrip()
        i=i.lstrip()
    #print(i)
        m.append(i)
    n=[]

    for i in range(1,len(m),2):
        j=m[i-1]+" | "+m[i]
        #print(j)
        n.append(j)

    o=[]
    for i in n:
        y=i.split(' ')
        j=y[1]+" "+y[4]+" "+y[7]
        o.append(j)
    #print(o)

    q=[]
    for i in o:
        z=i.split(" ")
        q.append(z)
    #print(q)
    silent_segments=[]
    nonsilent_segments=[]
    count=0
    #take silent segments>5.00s
    while count < len(q):
        segment = q[count]
        #print(segment)
        dur = segment[2]
        if(float(dur) > 5.00):
            silent_segments.append(segment)
        count=count+1
    print(len(silent_segments))
    #compute non-silent segments
    cnt=1
    while cnt < len(silent_segments):
        nonsilent_segment=[]
        segment = silent_segments[cnt]
        prev_segment = silent_segments[cnt-1]
        #print(segment)
        end_dur_prev = prev_segment[1]
        start_dur_curr = segment[0]
        if(float(end_dur_prev)< float(start_dur_curr)):
            nonsilent_segment.append(end_dur_prev)
            nonsilent_segment.append(start_dur_curr)
            dur = float(start_dur_curr)-float(end_dur_prev)
            nonsilent_segment.append(str(dur))
            nonsilent_segments.append(nonsilent_segment)
        cnt=cnt+1
    #get  the last segment until end of duration
    if len(silent_segments)!=0: 
        last_silent_segment = silent_segments[len(silent_segments)-1]
        #print(last_silent_segment)
        #print(duration)
        extra_segment=[]
        if(float(last_silent_segment[1])<duration):
            extra_segment.append(last_silent_segment[1])
            extra_segment.append(duration)
            extra_dur = float(last_silent_segment[1])-float(duration)
            extra_segment.append(str(dur))
            nonsilent_segments.append(extra_segment)
    #get the first segment in case of single segment
    if len(silent_segments)==1:
        seg=silent_segments[0]
        if(seg[0]!=0):
            nonsilent_segments.append(['0.00',seg[0],seg[0]])
    print("----Silent Segments----")
    print(silent_segments)
    print("----Non silent Segments----")
    print(nonsilent_segments)
    all_segments = silent_segments+nonsilent_segments
    #print(all_segments)
    print(len(all_segments))
    if len(all_segments)==0:
        os.rename(VideoTitle,"segments1.mp4")
    else:
        with open('manifest.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["start_time", "length", "rename_to"])
            k=0
            for i in silent_segments:
                a=float(i[0])
                dur=float(i[2])
                k=k+1
                writer.writerow([a,dur,"silent_segments"+str(k)])
            k=0
            for i in nonsilent_segments:
                a=float(i[0])
                dur=float(i[2])
                k=k+1
                writer.writerow([a,dur,"nonsilent_segments"+str(k)])

        split_by_manifest(VideoTitle,"manifest.csv")

def transcript_extract(storage_uri,j,segment_transcripts,segment_durations):
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
    segment_transcripts.append([j,segment_durations[j-1][0],text,segment_durations[j-1][1],len(nltk.sent_tokenize(text)),word_count])


def RankSegments():
    print("-------------Ranking Segments to get Features and Transcript extraction-------------")
    path = os.getcwd()
    files = []
    for r, d, f in os.walk(path):
        for file in f:
           if re.match(r'nonsilent_segments[0-9]*.mp4',file):
                files.append(file)    
    print(files)            
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
    print(wavfiles)         
    # os.system('export GOOGLE_APPLICATION_CREDENTIALS="tts.json"')            
    for fp in wavfiles:
        storage_client = storage.Client()
        bucket = storage_client.bucket('final_sem3')
        blob = bucket.blob(fp)
        blob.upload_from_filename(fp)
        print(
            "File {} uploaded to {}.".format(
                fp, fp
            )   
        )  
    segment_durations = []
    for i in files:
        cap = cv2.VideoCapture(i)
        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        segment_durations.append([i,duration])   
    segment_transcripts = []
    j=1    
    for k in range(1,len(wavfiles)+1):
            transcript_extract('gs://final_sem3/'+str(k)+'.wav',j,segment_transcripts,segment_durations)
            j=j+1        
    features = segment_transcripts
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
      i[0] = int(s[3]) 
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

def TextSummaryExtractor(VideoTitle,TranscriptFile):
    print("-------------Text Summary Extractor-------------")
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
    heading = VideoTitle.replace(".mp4"," ")
    summary = '\033'+heading+':'+'\n'+' '
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

def CondenseSegments():
    print("-------------Condensing the VIDEO---------------------")
    nonsilent_segments = [vi for vi in glob.glob("nonsilent_segments*.mp4")]
    print(nonsilent_segments)
    final_clip = VideoFileClip(nonsilent_segments[0],audio=True)
    for i in range(1,len(nonsilent_segments)):
        clip = VideoFileClip(nonsilent_segments[i],audio=True)
        final_clip = concatenate_videoclips([final_clip,clip],method="compose")
    final_clip.write_videofile("condensed.mp4")


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

def pdf_generator(path):
    print("-----------------Generating PDF--------------------------")
    chalks = [ori for ori in glob.glob("[0-9].jpg")]
    chalks = chalks + [ori for ori in glob.glob("[0-9][0-9].jpg")]
    chalks = chalks + [ori for ori in glob.glob("[0-9][0-9][0-9].jpg")]
    print(chalks)
    chalks = [s.replace(".jpg","") for s in chalks]
    chalks.sort(key=int)
    with open(path+'/output_summary.txt', 'r') as content_file:
            content = content_file.read()   
    breaks=content.split('.')
    pdf = FPDF()
    final_content=""
    print(chalks)
    for k in breaks:
        final_content=final_content+k+"."+"\n"
    #print(final_content)
    pdf.add_page()
    pdf.set_font('Times', '', 14)
    pdf.multi_cell(0, 5, final_content)
    pdf.ln()
    for j in chalks:
        image_path=path+"/"+str(j)+".jpg"
        pdf.add_page()
        pdf.image(image_path, x=10, y=8, w=100)
    pdf.output("summary.pdf")

def extractImportantChalkBoardRepresentation():
    print("-----------------Generating Chalkboard Frames--------------------------")
    framegen()
    chalk()
    refine_chalks()

def extractImportantPPTRepresentations():
    print("-----------------Generating Presentation Slides Frames--------------------------")
    framegen()
    ppt()
    refine_ppt()


def generatePDF():
    pdf_generator(os.getcwd())

def main(videoTitle, option):
    getAudioFromVideo(videoTitle)
    SegmentVideo(videoTitle)
    RankSegments()
    TextSummaryExtractor(videoTitle,'transcript.txt')
    CondenseSegments()
    if(option == '1'):
        extractImportantChalkBoardRepresentation()
    elif(option == '2'):
        extractImportantPPTRepresentations() 
    else:
        print("---Wrong option, exiting!!!!---")
        exit(0)
    generatePDF()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2]);    