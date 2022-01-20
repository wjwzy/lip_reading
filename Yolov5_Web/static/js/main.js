$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        var objUrl = getObjectURL(this.files[0]) ;
             console.log("objUrl = "+objUrl) ;
             if (objUrl) {
                 $("#video0").attr("src", objUrl) ;
             }

        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

     //建立一个可存取到该file的url
     function getObjectURL(file) {
         var url = null ;
         if (window.createObjectURL!=undefined) { // basic
             url = window.createObjectURL(file) ;
         } else if (window.URL!=undefined) { // mozilla(firefox)
             url = window.URL.createObjectURL(file) ;
         } else if (window.webkitURL!=undefined) { // webkit or chrome
             url = window.webkitURL.createObjectURL(file) ;
         }
         return url ;
     }

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' 识别结果:  ' + data);
                console.log('Success!');
            },
        });
    });

});
