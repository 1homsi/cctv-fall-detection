const input = document.getElementsByName("iterations");

const py_video = () => {
    eel.video_feed()();
};

const closeConnection = () => {
    eel.Close()();
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