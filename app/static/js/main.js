CAPTURE_IMG_WIDTH = 640
CAPTURE_IMG_HEIGHT = 480

// Show the selected image to the UI before uploading
function readURL(input, id) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    
    reader.onload = function(e) {
      $(id).attr('src', e.target.result).css({'width': CAPTURE_IMG_WIDTH, 'height': CAPTURE_IMG_HEIGHT});
    }
    
    reader.readAsDataURL(input.files[0]);
  }
}

// jQuery.ajaxSetup({
//   beforeSend: function() {
//      $('#loading').removeClass('hidden');
//   },
//   complete: function(){
//      $('#loading').addClass('hidden');
//   },
//   success: function() {
//     $('#loading').addClass('hidden');
//   }
// });

function initialize_webcam() {
  // HTML5 WEBCAM
  Webcam.set({
    width: CAPTURE_IMG_WIDTH,
    height: CAPTURE_IMG_HEIGHT,
    image_format: 'jpeg',
    jpeg_quality: 90
  });
  Webcam.attach( '#my-camera' );
  $('.camera-guide, .camera-wrapper').css({width: CAPTURE_IMG_WIDTH, height: CAPTURE_IMG_HEIGHT})
}

let form_capture = document.getElementById('form-capture-image')
$('.btn-capture-image').on('click', function(e) {
  e.preventDefault();

  $(this).addClass('is-loading');

  Webcam.snap(function(data_uri) {
    // display results in page
    let json_data = {'data-uri': data_uri }
    let camera = $('#my-camera')
    $('#my-camera').addClass('hidden');
    $('.taken-photo').attr('src', data_uri).removeClass('hidden');

    $.ajax({
      type: 'POST',
      url: '/upload/',
      processData: false,
      contentType: 'application/json; charset=utf-8',
      dataType: 'json',
      data: JSON.stringify(json_data),
      success: function(data) {
        $('#my-camera').removeClass('hidden');
        $('.results').removeClass('hidden')
        $('.taken-photo').attr('src', data_uri).addClass('hidden');
        $('.btn-capture-image').removeClass('is-loading')
        $('.face-found').text(data['face_found_in_image'])
        $('.is-obama').text(data['is_picture_of_obama'])
      }
    });
  });
});

$(document).ready(function() {
  initialize_webcam();
});

