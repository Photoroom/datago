use datago::DatagoClient;
use serde_json::json;
use std::io::Cursor;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::time::timeout;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const WAIT: Duration = Duration::from_secs(5);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reaching_the_limit_cancels_the_next_page_without_another_read() {
    let listing = MockServer::start().await;
    let payloads = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = payloads.local_addr().unwrap();
    Mock::given(method("GET"))
        .and(path("/images/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{
                "id": "first", "source": "fixture", "attributes": {},
                "image_direct_url": format!("http://{address}/sample.png")
            }],
            "next": format!("http://{address}/next")
        })))
        .expect(1)
        .mount(&listing)
        .await;

    // This integration-test process is independent of the live DB tests.
    std::env::set_var("DATAROOM_API_URL", format!("{}/", listing.uri()));
    std::env::set_var("DATAROOM_API_KEY", "fixture");
    let client = DatagoClient::new(
        json!({
            "source_type": "db",
            "source_config": {"sources": "fixture", "page_size": 1},
            "limit": 1, "samples_buffer_size": 1,
        })
        .to_string(),
    );
    client.start();

    let (mut image, mut next_page) = timeout(WAIT, async {
        let mut image = None;
        let mut next_page = None;
        for _ in 0..2 {
            let (connection, _) = payloads.accept().await.unwrap();
            let mut request = BufReader::new(connection);
            let mut line = String::new();
            request.read_line(&mut line).await.unwrap();
            let target = line.split_ascii_whitespace().nth(1).unwrap().to_owned();
            while line != "\r\n" {
                line.clear();
                assert_ne!(request.read_line(&mut line).await.unwrap(), 0);
            }
            match target.as_str() {
                "/sample.png" => image = Some(request),
                "/next" => next_page = Some(request),
                other => panic!("unexpected request for {other}"),
            }
        }
        (image.unwrap(), next_page.unwrap())
    })
    .await
    .expect("expected the image and next-page requests");

    // The worker can reach its limit only after this response. The next page
    // stays idle so it can finish only when the client cancels it.
    let mut png = Cursor::new(Vec::new());
    image::RgbImage::new(4, 3)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let png = png.into_inner();
    image
        .write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                png.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    image.write_all(&png).await.unwrap();

    let sample = client.try_get_sample_timeout(WAIT).unwrap().unwrap();
    assert_eq!(sample.id, "first");
    let mut trailing = [0];
    assert_eq!(
        timeout(WAIT, next_page.read(&mut trailing))
            .await
            .expect("the next page was still open after reaching the limit")
            .unwrap(),
        0
    );
    assert!(client.try_get_sample_timeout(WAIT).unwrap().is_none());
}
