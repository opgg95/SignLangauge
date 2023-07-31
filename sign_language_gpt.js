function getVideo() {
  if (!isVideoPlaying) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function(stream) {
        videoPlayer.srcObject = stream;
        videoPlayer.play();
        isVideoPlaying = true;
      })
      .catch(function(error) {
        console.error('Webcam not working', error);
      });
  } else {
    videoPlayer.pause();
    videoPlayer.srcObject = null;
    isVideoPlaying = false;
  }
}

var videoButton = document.querySelector('button');
videoButton.addEventListener('click', function() {
  getVideo();
});

