import time
import threading
import sys
import StringIO
from StringIO import StringIO

class Suppress:
    """
    Suppresses output until either end() is called or it goes out of scope.
    Should be used with context manager.
    """
    def __init__(self):
        self.oldout = sys.stdout
        sys.stdout = StringIO()

    def __del__(self):
        self.end()

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_value, traceback):
        self.end()

    def end(self):
        sys.stdout = self.oldout

class Progress(object):
    """ Keeps track of progress of code. """
    def __init__(self, size, length=50, eta=True, printing=True):
        """ This starts the progress bar running.

        @param size: the total number of tasks that need to be run
        @type size: float

        @param length: the size of the bar on the screen
        @type length: int

        @param eta: do we show an eta?
        @type eta: boolean

        @param printing: Do we print?
        @type printing: boolean
        """
        self.status = "Running..."
        self.printing = printing
        self.finished = False
        self.size = size
        self.length = length
        self.prev = 0
        self.eta = eta
        self.start = time.time()
        self.update(0)

    def update(self, progress, status="Running..."):
        """
        Update the progress bar to be equal to a certain amount of progress.

        @param progress: the new amount of progress that we have (number)
        @type progress: float

        @param status: status to show, if anything.
        @type status: str
        """
        self.prev = progress
        normalized = min(progress/float(self.size),1)
        block = int(round(self.length*(normalized)))
        if self.printing:
            if self.eta:
                sys.stdout.write(
                    "\rProgress: [{0}] {1}% {2}. {3}".format( 
                        "#"*block + "-"*(self.length-block), "%.2f" \
                        % round(normalized*100,2), "Left: %d" \
                    % int((time.time()-self.start)*(1-normalized)/(
                        0.00000001+normalized)), str(status)))
            else:
                sys.stdout.write("\rProgress: [{0}] {1}% {2}".format( 
                    "#"*block + "-"*(self.length-block), round(normalized*100,
                        2), status))
            sys.stdout.flush()

    def increment(self, increment, status="Running..."):
        """
        Increase the total amount of progress by the increment. 

        @param increment: how much to increase our progress. Can be float.
        @type increment: float

        @param status: status to show, if anything. 
        @type status: str
        """
        self.update(self.prev+increment, status)

    def finish(self):
        """ Stop running the progress bar. """
        if self.printing:
            sys.stdout.write("\rProgress: [{0}] {1}% {2}\n\n".format( 
                "#"*self.length , 100, "Time Taken: "+str(int(time.time()
                    -self.start))+(" "*100)))
        self.finished = True