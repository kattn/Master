mapboxgl.accessToken = 'pk.eyJ1IjoiYmplcmtlIiwiYSI6ImNqbHJxbDh1bjA3YjczcXAwdnN2dmN3eGgifQ.aHKfckk6rhXMGSZiw9tfLA';
var map = new mapboxgl.Map({
    container: 'map', // container id
    style: 'mapbox://styles/mapbox/streets-v9', // stylesheet location
    center: [11.191023, 60.148844], // starting position [lng, lat]
    zoom: 12 // starting zoom
});

loadPrincepality();

function loadPrincepality() {
    var xojb = new XMLHttpRequest();
    xojb.overrideMimeType("application/json");
    xojb.open('GET', "http://127.0.0.1:8080", true);
    xojb.onreadystatechange = function (ev) {
        if (xojb.readyState === 4 && xojb.status === "200") {
                var json = JSON.parse(xojb.responseText);
                console.log(json);
        }
    };
    xojb.send(null);
}
