const input = document.getElementsByName("iterations");

const py_video = () => {
    eel.video_feed()();
};

const py_video_fall = () => {
    eel.detectFallFeed("0")();
};

const closeConnection = () => {
    eel.Close()();
};

const closeFallConnection = () => {
    eel.CloseDetector()();
};

const TrainModel = async () => {
    let info = document.getElementById("info");
    info.style.visibility = "visible";
    var n = await eel.train(input[0]?.value)();
    info.innerHTML = n;
};

eel.expose(updateImageSrc);
function updateImageSrc(val) {
    let elem = document.getElementById("bg");
    elem.src = "data:image/jpeg;base64," + val;
}


