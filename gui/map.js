mapboxgl.accessToken = 'pk.eyJ1IjoiYmplcmtlIiwiYSI6ImNqbHJxbDh1bjA3YjczcXAwdnN2dmN3eGgifQ.aHKfckk6rhXMGSZiw9tfLA';
var map = new mapboxgl.Map({
    container: 'map', // container id
    style: 'mapbox://styles/mapbox/streets-v9', // stylesheet location
    center: [11.191023, 60.148844], // starting position [lng, lat]
    zoom: 12 // starting zoom
});

var WebSocket = WebSocket;
var connection = null;
var connected = false;
var features = null;

connect();

function loadPrincepality() {
    connection.send(JSON.stringify({
        "command" : "REQUEST_FEATURE",
        "content" : ""
    }));
}

function connect() {
    var serverUrl = "ws://127.0.0.1:8080";

    connection = new WebSocket(serverUrl);

    connection.onopen = function(ev) {
        connected = true;
        loadPrincepality();
    }

    connection.onclose = function(ev) {
        connected = false;
    }

    connection.onmessage = function(ev) {
        features = JSON.parse(ev.data).content;
        console.log(features);

        map.addLayer({
            "id": features[0].name,
            "type": "circle",
            "source": {
                "type": "geojson",
                "data": features[0]
            },
            "paint": {
                "circle-radius": 3,
                "circle-color": "#007cbf"
            }
        });

        map.addLayer({
            "id": features[1].name,
            "type": "circle",
            "source": {
                "type": "geojson",
                "data": features[1]
            },
            "paint": {
                "circle-radius": 3,
                "circle-color": "#007cbf"
            }
        });

        map.addLayer({
            "id": features[2].name,
            "type": "line",
            "source": {
                "type": "geojson",
                "data": features[2]
            },
            "paint": {
                "line-width": 3,
                "line-color": "#007cbf"
            }
        });
    }
}
