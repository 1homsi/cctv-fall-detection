const input = document.getElementsByName("iterations");


window.onload = () => {
    window.resizeTo(800, 600);
    window.onresize = () => {
        window.resizeTo(800, 600);
    };
};


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


function handleMenu(e) {
    let menu = document.getElementById("Menu");
    let open = document.getElementById("open");
    let close = document.getElementById("close");
    if (menu.style.display === "block") {
        menu.style.display = "none";
        open.style.display = "block";
        close.style.display = "none";
    } else {
        menu.style.display = "block";
        open.style.display = "none";
        close.style.display = "block";
    }
};
