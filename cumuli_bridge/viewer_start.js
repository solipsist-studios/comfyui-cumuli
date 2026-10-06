// Starts the pinned Rerun web viewer on one recording. Served by the pack's own route, next to
// the viewer files, so it needs no cross-origin access and no inline script (the page's
// Content-Security-Policy does not allow either).

const message = document.getElementById("message");
const params = new URLSearchParams(location.search);

function fail(text) {
    message.textContent = text;
    message.style.display = "grid";
}

// The viewer misreads a bare path ("/cumuli/viewer/rrd/x.rrd" became the host "cumuli"), so it is
// given an absolute URL. Only this origin is allowed: the page's policy would block anything else
// anyway, and saying so is clearer than a silent network error.
let recording = null;
try {
    const given = params.get("rrd");
    if (given) {
        const absolute = new URL(given, location.href);
        recording = absolute.origin === location.origin ? absolute.toString() : null;
    }
} catch (error) {
    recording = null;
}

if (!recording) {
    fail("No usable recording was given: the page plays a recording from its own server only.");
} else {
    try {
        const { WebViewer } = await import("./assets/index.js");
        const viewer = new WebViewer();
        await viewer.start(recording, document.getElementById("viewer"), {
            hide_welcome_screen: true,
            width: "100%",
            height: "100%",
            allow_fullscreen: false,   // the page sits in an iframe; the node owns its size
        });
        message.style.display = "none";
    } catch (error) {
        console.error("cumuli viewer:", error);
        fail(`The viewer could not start: ${error?.message ?? error}`);
    }
}
