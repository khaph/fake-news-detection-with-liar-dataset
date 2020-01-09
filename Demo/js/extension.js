// document.addEventListener('DOMContentLoaded', function() {
//     document.getElementById('status').textContent = "Extension loaded";
//     var button = document.getElementById('changelinks');
//     button.addEventListener('click', function () {
//         $('#status').html('Clicked change links button');
//         var text = $('#linkstext').val();
//         if (!text) {
//             $('#status').html('Invalid text provided');
//             return;
//         }
//         chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
//             chrome.tabs.sendMessage(tabs[0].id, {data: text}, function(response) {
//                 $('#status').html('changed data in page');
//                 console.log('success');
//             });
//         });
//     });
// });

// document.onmouseup = function(e){
//     if (window.getSelection().toString() != ""){
//         console.log(window.getSelection().toString())
//         console.log(e.clientX, e.clientY)
//         var x = document.createElement("div")
//         x.innerHTML = "kha"
//         x.style.left = e.clientX + "px"
//         x.style.top = e.clientY + "px"
//         x.style.position = "fixed"
//         document.body.appendChild(x)
//     }
// }