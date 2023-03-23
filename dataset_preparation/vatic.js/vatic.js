"use strict";

class FramesManager {
  constructor() {
    this.frames = {
      totalFrames: () => { return 0; }
    };
    this.onReset = [];
  }

  set(frames) {
    this.frames = frames;
    for (let i = 0; i < this.onReset.length; i++) {
      this.onReset[i]();
    }
  }
}

/**
 * Extracts the frame sequence of a video file.
 */
function extractFramesFromVideo(video, canvas, config, file) {
  let resolve = null;
  let ctx = canvas.getContext('2d');
  let totalFrames = 0;
  let duration = 0;
  let totalSteps = 0;

  return new Promise((_resolve, _) => {
    resolve = _resolve;

    function onload() {
      duration = video.duration;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // console.log(video.duration);

      // video.addEventListener('seeked', function() {
      //     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // }, false);

      video.currentTime=0.0000001;
      video.currentTime=0;

      totalSteps = duration / config.stepSize;  // show one frame each 0.1 seconds

      video.playbackRate = config.playbackRate;

      resolve({
        width: video.videoWidth,
        height: video.videoHeight,
        totalFrames: () => { return Math.ceil(totalSteps); },
        getFrame: (frameNumber) => {
          return new Promise((resolve, _) => {resolve();});
        },
        seekVideoAndGetFrame: (frameNumber) => {
          video.currentTime = frameNumber * config.stepSize;
          return new Promise((resolve, _) => {resolve();});
        }
      });
    }

    video.src = URL.createObjectURL(file);
    video.addEventListener('loadeddata', onload, false);

  });
}

/**
 * Extracts the frame sequence from a previously generated zip file.
 */
function extractFramesFromZip(config, file) {
  return new Promise((resolve, _) => {
    JSZip
      .loadAsync(file)
      .then((zip) => {
        let totalFrames = 0;
        for (let i = 0; ; i++) {
          let file = zip.file(i + config.imageExtension);
          if (file == null) {
            totalFrames = i;
            break;
          }
        }

        // console.log(this);

        resolve({
          totalFrames: () => { return totalFrames; },
          getFrame: (frameNumber) => {
            // console.log(this);
            return new Promise((resolve, _) => {
              let file = zip.file(frameNumber + config.imageExtension);
              file
                .async('arraybuffer')
                .then((content) => {
                  let blob = new Blob([ content ], {type: config.imageMimeType});
                  resolve(blob);
                });
            });
          }
        });
      });
  });
}

/**
 * Represents the coordinates of a bounding box
 */
class BoundingBox {
  constructor(x, y, width, height) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
}

/**
 * Represents a bounding box at a particular frame.
 */
class AnnotatedFrame {
  constructor(frameNumber, bbox, isGroundTruth, visible=true, behaviour='other') {
    this.frameNumber = frameNumber;
    this.bbox = bbox;
    this.isGroundTruth = isGroundTruth;
    this.visible = visible;
    this.behaviour = behaviour;
  }

  isVisible() {
    return this.visible;
  }
}

/**
 * Represents an object bounding boxes throughout the entire frame sequence.
 */
class AnnotatedObject {
  constructor() {
    this.frames = [];
    this.name = null;
    this.id = null;
  }

  add(frame) {
    for (let i = 0; i < this.frames.length; i++) {
      if (this.frames[i].frameNumber == frame.frameNumber) {
        this.frames[i] = frame;
        this.removeFramesToBeRecomputedFrom(i + 1);
        return;
      } else if (this.frames[i].frameNumber > frame.frameNumber) {
        this.frames.splice(i, 0, frame);
        this.removeFramesToBeRecomputedFrom(i + 1);
        return;
      }
    }

    this.frames.push(frame);
  }

  get(frameNumber) {
    for (let i = 0; i < this.frames.length; i++) {
      let currentFrame = this.frames[i];
      if (currentFrame.frameNumber > frameNumber) {
        break;
      }

      if (currentFrame.frameNumber == frameNumber) {
        return currentFrame;
      }
    }

    return null;
  }

  getPrev(frameNumber) {
    for (let i = this.frames.length - 1; i >= 0; i--) {
      let currentFrame = this.frames[i];
      if (currentFrame.frameNumber < frameNumber) {
        return currentFrame;
      }
    }
    return null;
  }

  getNext(frameNumber) {
    for (let i = 0; i < this.frames.length; i++) {
      let currentFrame = this.frames[i];
      if (currentFrame.frameNumber > frameNumber) {
        return currentFrame;
      }
    }

    return null;
  }

  removeFramesToBeRecomputedFrom(frameNumber) {
    let count = 0;
    for (let i = frameNumber; i < this.frames.length; i++) {
      if (this.frames[i].isGroundTruth) {
        break;
      }
      count++;
    }
    if (count > 0) {
      this.frames.splice(frameNumber, count);
    }
  }


}

/**
 * Tracks annotated objects throughout a frame sequence using optical flow.
 */
class AnnotatedObjectsTracker {
  constructor(framesManager) {
    this.framesManager = framesManager;
    this.annotatedObjects = [];
    this.lastFrame = -1;
    this.ctx = document.createElement('canvas').getContext('2d');

    this.framesManager.onReset.push(() => {
      this.annotatedObjects = [];
      this.lastFrame = -1;
    });
  }

  getFrameWithObjects(frameNumber) {
    return new Promise((resolve, _) => {
      // let i = this.startFrame(frameNumber);

      this.framesManager.frames.getFrame(frameNumber).then(() => {
          let result = [];
          // let toCompute = [];
          for (let i = 0; i < this.annotatedObjects.length; i++) {
            let annotatedObject = this.annotatedObjects[i];
            let annotatedFrame = annotatedObject.get(frameNumber);
            if (annotatedFrame == null) {
              // do we have a prev and next? then interpolate

              // annotatedFrame = annotatedObject.get(frameNumber - 1);
              // if (annotatedFrame == null) {
              //   throw 'tracking must be done sequentially';
              // }
              // toCompute.push({annotatedObject: annotatedObject, bbox: annotatedFrame.bbox});
            } else {
              result.push({annotatedObject: annotatedObject, annotatedFrame: annotatedFrame});
            }
          }
          // console.log(this.annotatedObjects);
          // console.log(result);

          resolve({img: null, objects: result});
      });

    });
  }

  // startFrame(frameNumber) {
  //   for (; frameNumber >= 0; frameNumber--) {
  //     let allObjectsHaveData = true;
  //
  //     for (let i = 0; i < this.annotatedObjects.length; i++) {
  //       let annotatedObject = this.annotatedObjects[i];
  //       if (annotatedObject.get(frameNumber) == null) {
  //         allObjectsHaveData = false;
  //         break;
  //       }
  //     }
  //
  //     if (allObjectsHaveData) {
  //       return frameNumber;
  //     }
  //   }
  //
  //   throw 'corrupted object annotations';
  // }

  imageData(img) {
    let canvas = this.ctx.canvas;
    canvas.width = img.width;
    canvas.height = img.height;
    this.ctx.drawImage(img, 0, 0);
    return this.ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
};
