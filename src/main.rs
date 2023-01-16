use opencv::{
    prelude::*,
    highgui,
    imgproc::{self, LINE_8}, core::{CV_8U, Point2i, BORDER_CONSTANT,  Vector, Scalar, find_non_zero, Point}, 
};

type VectorOfPoint = Vector<Point>;

fn mean_point(points: VectorOfPoint) -> Option<Point> {
    
    if points.len() > 0 {

        let mut result = Point::new(0, 0);
    
        for point in &points {
            result.x += point.x;
            result.y += point.y;
        }

        result.x /= points.len() as i32;
        result.y /= points.len() as i32;

        Some(result)
    } else {
        None
    }
}

fn main() {
    // Before using any pylon methods, the pylon runtime must be initialized.
    let pylon = pylon_cxx::Pylon::new();

    // Create an instant camera object with the camera device found first.
    let camera = pylon_cxx::TlFactory::instance(&pylon).create_first_device().unwrap();

    // Print the model name of the camera.
    println!("Using device {:?}", camera.device_info().model_name().unwrap());

    camera.open().unwrap();

    // camera.enum_node("PixelFormat")?.set_value("RGB8")?;

    match camera.node_map().integer_node("GevSCPSPacketSize") {
        Ok(mut node) => node.set_value(9000).unwrap(),
        Err(e) => eprintln!("Ignoring error getting PacketSize node: {}", e),
    };

    match camera.node_map().integer_node("GevSCPSPacketSize") {
        Ok(node) => println!(
            "Packet Size: {}",
            node.value().unwrap_or(0)
        ),
        Err(e) => eprintln!("Ignoring error getting PacketSize node: {}", e),
    };

    match camera.node_map().enum_node("PixelFormat") {
        Ok(node) => println!(
            "pixel format: {}",
            node.value().unwrap_or("could not read value".to_string())
        ),
        Err(e) => eprintln!("Ignoring error getting PixelFormat node: {}", e),
    };

    // Start the grabbing of COUNT_IMAGES_TO_GRAB images.
    // The camera device is parameterized with a default configuration which
    // sets up free-running continuous acquisition.
    camera.start_grabbing(&pylon_cxx::GrabOptions::default()).unwrap();

    let mut grab_result = pylon_cxx::GrabResult::new().unwrap();

    highgui::named_window("window", highgui::WINDOW_NORMAL).unwrap();

    let mut trigger = false;

    let mut hits = VectorOfPoint::new();

    while camera.is_grabbing() {
        // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        camera.retrieve_result(
            5000,
            &mut grab_result,
            pylon_cxx::TimeoutHandling::ThrowException,
        ).unwrap();

        // Image grabbed successfully?
        if grab_result.grab_succeeded().unwrap() {
            // Access the image data.
            //println!("SizeX: {}", grab_result.width().unwrap());
            //println!("SizeY: {}", grab_result.height().unwrap());

            let image_buffer = grab_result.buffer().unwrap();
            //println!("Value of first pixel: {}\n", image_buffer[0]);

            let mut img = Mat::from_slice(image_buffer).unwrap().reshape(1, grab_result.height().unwrap() as i32).unwrap();

            let mut threshold_img = Mat::default();
            let mut morph_img = Mat::default();
            let kernel = Mat::ones(3, 3, CV_8U).unwrap();
            imgproc::threshold(&img, &mut threshold_img, 200.0, 255.0, imgproc::THRESH_BINARY).unwrap();
            imgproc::morphology_ex(&threshold_img, &mut morph_img, imgproc::MORPH_CLOSE, &kernel, Point2i::new(-1, -1), 3, BORDER_CONSTANT,imgproc::morphology_default_border_value().unwrap()).unwrap();

            let mut points= VectorOfPoint::new();

            find_non_zero(&morph_img, &mut points).unwrap();

            let point = mean_point(points);

            if let Some(p) = point {
                imgproc::circle(&mut img, p, 20, Scalar::new(255.0, 255.0, 255.0, 255.0), 1, LINE_8, 0).unwrap();
                if  trigger == false {
                    println!("Hit {} {}", p.x, p.y);
                    hits.push(p);
                    trigger = true;
                }
            } else {
                trigger = false;
            }

            for p in &hits {
                imgproc::circle(&mut img, p, 5, Scalar::new(255.0, 255.0, 255.0, 255.0), 1, LINE_8, 0).unwrap();
            }

            highgui::imshow("window", &img).unwrap();
            let key = highgui::poll_key().unwrap() as u8 as char;

            match key {
                'q' => camera.stop_grabbing().unwrap(),
                'r' => hits = VectorOfPoint::new(),
                _ => (),
            }

        } else {
            println!(
                "Error: {} {}",
                grab_result.error_code().unwrap(),
                grab_result.error_description().unwrap()
            );
        }
    }
}
