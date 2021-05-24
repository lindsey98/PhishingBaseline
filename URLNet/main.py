import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import threading
from datetime import datetime
import os
import subprocess
import multiprocessing
import queue
import urllib.request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import ssl
from threading import Lock
import utils_urlnet
import urlnet_config as config

import sys

date_time = datetime.now().strftime("%Y%m%d")
date_time_catcher = datetime.now().strftime("%Y-%m-%d")

suspicious_domains_output = "suspicious_domains_" + date_time_catcher + ".log"

# Global variable to terminate all threads in one shot
lock = Lock()
global threads_kill
lock.acquire()
threads_kill = False
lock.release()

# Queue is to handle between suspicious_logs and clean url net
q = queue.Queue()

# Queue is to handle between urlnet clean and urlnet retrieve
q1 = queue.Queue()

# Queue is to handle urlnet results and sending to screenshots thread
q2 = queue.Queue()


###############################################################################
# Methods to deal with watchdog
###############################################################################
def createEventHandler():
    patterns = "*"
    ignore_patterns = ""
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_modified = on_modified

    return my_event_handler

def create_observer(event_handler):
    path = "../"
    go_recursively = True
    my_observer = Observer()    
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    return my_observer

# Monitor the creation of suspicious log file
def on_created(event):
    if "urlnet" in event.src_path:
        print(f"{event.src_path} has been created!")

################################################################################
# Whenever suspicious_logs get modified, add to queue
# Whenever urlnet_cleaned gets updated, add to queue for input to URLNet
# Whenever urlnet_phish gets updated, check if its 1 (phishing), if its predicted as phishing, send to SS
################################################################################
def on_modified(event):
    if "suspicious_domains" in event.src_path:
        with open(event.src_path, "r") as f:
            q.put((list(f)[-1]))

    # Need to change this portions to variables instead of hardcoding the names 
    # Use a utils script file to store all these constant names
    # queue.get() is a blocking function, therefore, it has its own internal lock
    # Places a tuple (label, url) into the queue for retrieval for testing with URLNet
    elif "urlnet_cleaned" in event.src_path and "urlnet" in event.src_path:
        with open(event.src_path, "r") as f:
            last_line = list(f)[-1].split('\t')
            label = last_line[0]
            url = last_line[1].strip()
            q1.put((url, label))

    elif "urlnet_phish" in event.src_path and "urlnet" in event.src_path:
        with open(event.src_path, "r") as f:
            last_line = list(f)[-1].split('\t')
            urlnet_prediction = last_line[2]
            url_screenshot = last_line[0]
            if isinstance(urlnet_prediction, int):
                if int(round(urlnet_prediction)) == 1:
                    q2.put(url_screenshot)


def start_observer(my_observer):
    my_observer.start()
    try:
        while not threads_kill:
            time.sleep(1)
            break
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()

        
###############################################################################
# Methods for other main tasks
###############################################################################
# def catch_phishing():
#     # Deciding how long the certstream should run before coming to a halt
#     catch_phishing_command = ["python", config.phish_catcher]
#     process1= subprocess.Popen(catch_phishing_command, shell=False).wait()

#     lock.acquire()
#     threads_kill = True
#     lock.release()
    
def add_scheme(domain):
    if not urlparse(domain).scheme:
        domain = "http://" + domain

    return domain

def test_url_alive(domain_intermediate, domain_set):
    clean_file = os.path.join(config.output_results, "urlnet_cleaned.txt")
    if "*" not in domain_intermediate:
        try:
            if(urllib.request.urlopen(domain_intermediate).getcode() == 200):
                with open(clean_file, 'a+') as f:
                    if domain_intermediate not in domain_set:
                        f.write("1\t")
                        f.write("{}\n".format(add_scheme(domain_intermediate))) 
        except Exception:
            pass

        return domain_intermediate
        

def clean_suspicious_domains():    
    # Grab from queue whenever there is a new entry
    domain_set = set()
    while True:
        if threads_kill:
            break
        else:
            while not q.empty() and not threads_kill:
                domain = q.get()
                domain_intermediate = add_scheme(domain.strip())
                domain_clean = test_url_alive(domain_intermediate, domain_set)
                domain_set.add(domain_clean)
                q.task_done()

def urlnet():
    while True:
        if threads_kill:
            break
        else:
            while not q1.empty() and not threads_kill:
                url, label = q1.get()
                label = [label]
                url = [url]
                utils_urlnet.main(url,label)
                q1.task_done()


if __name__ == "__main__":
    # Creates an event handler to handle whenever an event happens
    my_event_handler = createEventHandler()
    # Creates an observer watch dog that utilizes the event handler to handle any changes to files

    # Thread for observer
    my_observer = create_observer(my_event_handler)
    p = threading.Thread(target=start_observer, args=(my_observer,))
    p.daemon=True
    p.start()

    # # Thread for certificate generations
    # p1 = threading.Thread(target=catch_phishing)
    # p1.daemon = True
    # p1.start()

    # Thread for cleaning these data
    p2 = threading.Thread(target=clean_suspicious_domains)
    p2.daemon=True
    p2.start()

    # Thread for running URLNet model
    p3 = threading.Thread(target=urlnet)
    p3.daemon=True
    p3.start()


    # Waiting for the threads to end to join back to the main thread
    p.join()
    # p1.join()   
    p2.join()
    p3.join()