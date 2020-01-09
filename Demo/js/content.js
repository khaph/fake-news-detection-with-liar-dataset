document.onmouseup = function(e){
    if (window.getSelection().toString() != ""){
        console.log(window.getSelection().toString())
        console.log(e.clientX, e.clientY)
        var x = document.createElement("img")
        x.src = chrome.runtime.getManifest().browser_action.default_icon
        x.style.left = e.clientX + "px"
        x.style.top = e.clientY + "px"
        x.style.position = "fixed"
        x.style.width = "200px"
        x.style.zIndex = "10000"
        document.body.appendChild(x)
        console.log(x)
    }
}