// Puts the Rerun viewer inside the "Cumuli Preview Ring (Rerun)" node.
//
// The node writes a recording and returns a token for it; this adds an <iframe> to the node that
// loads the pack's own viewer page (served from ComfyUI's port, under a Content-Security-Policy
// that allows no other origin) pointed at that recording. The viewer is a normal web page, so it
// keeps its own 3D navigation, timeline and WebGL context.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "CumuliPreviewRing";
const WIDGET = "cumuli_viewer";
const MIN_SIZE = [420, 380];

function viewerUrl(token) {
    const recording = api.apiURL(`/cumuli/viewer/rrd/${token}.rrd`);
    return api.apiURL(`/cumuli/viewer/page?rrd=${encodeURIComponent(recording)}`);
}

function viewerWidget(node, token) {
    const existing = node.widgets?.find((w) => w.name === WIDGET);
    if (existing) {
        if (existing.__token !== token) {
            existing.__token = token;
            existing.element.src = viewerUrl(token);
        }
        return existing;
    }
    const frame = document.createElement("iframe");
    frame.src = viewerUrl(token);
    // Same-origin, script-only: the viewer needs scripts and its own origin to fetch the
    // recording, and nothing else (no forms, popups, top navigation or downloads).
    frame.setAttribute("sandbox", "allow-scripts allow-same-origin");
    Object.assign(frame.style, { border: "0", borderRadius: "6px", width: "100%", height: "100%", background: "#0b0f10" });
    const widget = node.addDOMWidget(WIDGET, WIDGET, frame, {
        serialize: false,   // a viewer, not an input
        hideOnZoom: false,
    });
    widget.__token = token;
    if (node.size[0] < MIN_SIZE[0] || node.size[1] < MIN_SIZE[1]) {
        node.setSize([Math.max(node.size[0], MIN_SIZE[0]), Math.max(node.size[1], MIN_SIZE[1])]);
    }
    return widget;
}

app.registerExtension({
    name: "Cumuli.RerunViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const payload = message?.cumuli_view?.[0];
            if (!payload?.token) return;
            viewerWidget(this, payload.token);
            this.setDirtyCanvas(true, true);
        };
    },
});
